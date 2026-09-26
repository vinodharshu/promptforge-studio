import os
import secrets
import logging
from flask import Flask, send_from_directory, request, jsonify, session
from flask_cors import CORS
from importlib import import_module
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the optional extension dynamically
SQLAlchemy = import_module('flask_sqlalchemy').SQLAlchemy

app = Flask(__name__, static_folder='.', template_folder='.')

IS_PROD = os.getenv("FLASK_ENV", "production").lower() != "development"

# 1. Secret key setup
configured_secret = os.getenv("SECRET_KEY", "").strip()
if not configured_secret:
    configured_secret = secrets.token_hex(32)
    logger.warning("SECRET_KEY is not set. A temporary key is being used.")

app.secret_key = configured_secret

# 2. Session cookie settings
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = IS_PROD
app.config["SESSION_COOKIE_SAMESITE"] = (
    os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
    if not IS_PROD
    else os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
)

# 3. SQL DATABASE CONFIGURATION
db_url = os.getenv("DATABASE_URL", "").strip()

if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

if not db_url:
    logger.warning(
        "DATABASE_URL is missing. Using local SQLite database. "
        "Configure PostgreSQL on Render for persistent production storage."
    )
    db_url = "sqlite:///database.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

if db_url.startswith("postgresql://"):
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_recycle": 280,
    }

db = SQLAlchemy(app)

# 4. CORS Configuration
default_origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
]

configured_origins = os.getenv("FRONTEND_URL") or os.getenv("FRONTEND_ORIGIN") or ""
ALLOWED_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in configured_origins.split(",")
    if origin.strip()
]

ALLOWED_ORIGINS.extend(
    origin for origin in default_origins
    if origin not in ALLOWED_ORIGINS
)

CORS(
    app,
    supports_credentials=True,
    origins=ALLOWED_ORIGINS
)

@app.before_request
def _log_cross_origin_requests():
    origin = request.headers.get('Origin')
    if origin and origin not in ALLOWED_ORIGINS:
        print(f"⚠️ Request from origin '{origin}' is NOT in ALLOWED_ORIGINS")

@app.errorhandler(Exception)
def _handle_uncaught_exception(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return jsonify({'status': 'error', 'message': e.description}), e.code
    print(f"Unhandled exception: {e}")
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# Global GEMINI API KEY setup
raw_key = os.getenv("GEMINI_API_KEY", "")
DEFAULT_GEMINI_API_KEY = raw_key.strip().strip("'").strip('"') if raw_key else ""

# --- SQL MODELS ---
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    api_key = db.Column(db.String(255), default='')
    apps = db.relationship('App', backref='owner', lazy=True)

class App(db.Model):
    __tablename__ = 'apps'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    code = db.Column(db.Text, nullable=False)

def initialize_database():
    try:
        with app.app_context():
            db.create_all()
        logger.info("Database initialization completed.")
    except Exception:
        logger.exception(
            "Database initialization failed. "
            "The application will continue starting."
        )

initialize_database()

# --- FIXED GEMINI HELPER FUNCTION ---
def call_gemini_model(prompt_text, api_key=None):
    key_to_use = api_key if api_key else DEFAULT_GEMINI_API_KEY
    clean_key = key_to_use.strip().strip("'").strip('"')
    
    if not clean_key:
        raise ValueError("Gemini API key is missing.")

    # REST transport helps avoid some gRPC metadata issues.
    genai.configure(api_key=clean_key, transport="rest")

    # Current model order. You can override it in Render with GEMINI_MODEL.
    # Do not use retired 2.0/1.5 models here.
    configured_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    models_to_try = [configured_model]

    # Small fallback list for projects that do not yet have access to the newest model.
    for fallback in ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"):
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    last_exception = Exception("No Gemini model responded successfully.")

    for index, model_name in enumerate(models_to_try):
        try:
            logger.info("Trying Gemini model: %s", model_name)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt_text,
                request_options={"timeout": 75}
            )
            response_text = getattr(response, "text", None)
            if response_text and response_text.strip():
                return response_text.strip()

            last_exception = RuntimeError(
                f"Model {model_name} returned an empty response."
            )

        except Exception as e:
            last_exception = e
            logger.warning(
                "Gemini model %s failed: %s",
                model_name,
                e
            )
            # Continue only when a model is unavailable. For other errors,
            # trying multiple models can make Render hit its request timeout.
            error_text = str(e).lower()
            unavailable = any(term in error_text for term in (
                "not found", "not supported", "404", "unknown model", "invalid model"
            ))
            if not unavailable:
                break

    raise last_exception

# --- ROOT ROUTE ---
@app.route('/')
def index():
    if os.path.exists(os.path.join(app.root_path, 'templates', 'index.html')):
        return send_from_directory('templates', 'index.html')
    elif os.path.exists(os.path.join(app.root_path, 'index.html')):
        return send_from_directory('.', 'index.html')
    else:
        return "PromptForge Backend Service Running Successfully!", 200

# --- USER AUTHENTICATION ROUTES ---

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({'status': 'error', 'message': 'Username and password required!'}), 400

    try:
        if User.query.filter_by(username=username).first():
            return jsonify({'status': 'error', 'message': 'Username already exists!'}), 400

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)

        db.session.add(new_user)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'User registered successfully!'})
    except Exception as e:
        db.session.rollback()
        print(f"[/api/register] DB error: {e}")
        return jsonify({'status': 'error', 'message': 'Database error. Please try again shortly.'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    try:
        user = User.query.filter_by(username=username).first()
    except Exception as e:
        print(f"[/api/login] DB error: {e}")
        return jsonify({'status': 'error', 'message': 'Database error. Please try again shortly.'}), 500

    if user and check_password_hash(user.password, password):
        session['user_id'] = user.id
        session['username'] = user.username
        session.modified = True
        return jsonify({'status': 'success', 'username': user.username})
        
    return jsonify({'status': 'error', 'message': 'Invalid credentials!'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'status': 'success', 'message': 'Logged out successfully!'})

@app.route('/api/user-status', methods=['GET'])
def user_status():
    if 'user_id' in session:
        return jsonify({'logged_in': True, 'username': session['username']})
    return jsonify({'logged_in': False})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "ok",
        "database_configured": bool(os.getenv("DATABASE_URL")),
        "gemini_configured": bool(DEFAULT_GEMINI_API_KEY),
    }), 200

# --- GENERATE APP ROUTE ---

@app.route('/generate', methods=['POST'])
def generate():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Please login first!'}), 401

    data = request.get_json() or {}
    user_prompt = str(data.get('prompt') or '')
    previous_code = data.get('previous_code', '')
    custom_api_key = str(data.get('api_key') or '').strip()

    if not user_prompt:
        return jsonify({'status': 'error', 'message': 'Prompt is required!'}), 400

    active_api_key = custom_api_key if custom_api_key else DEFAULT_GEMINI_API_KEY

    if not active_api_key:
        return jsonify({'status': 'error', 'message': 'Gemini API Key missing!'}), 400

    try:
        system_instruction = (
            "You are a World-Class UI/UX & Web Developer.\n"
            "CRITICAL UPDATE RULE:\n"
            "- If 'PREVIOUS CODE' is provided, modify the EXISTING code.\n"
            "- DO NOT create a completely new topic or app.\n"
            "- ONLY apply requested changes on top of existing code.\n\n"
            "OUTPUT RULES:\n"
            "1. Return ONLY single-file valid HTML code with embedded CSS/JS.\n"
            "2. Do NOT wrap code in markdown. Return RAW HTML ONLY.\n"
            "3. Use Tailwind CSS via CDN inside <head>.\n"
        )

        full_prompt = f"{system_instruction}\n\nUSER PROMPT: {user_prompt}\n"
        if previous_code:
            full_prompt += f"\nPREVIOUS CODE TO MODIFY:\n{previous_code}"

        generated_code = call_gemini_model(full_prompt, api_key=active_api_key)

        if "```html" in generated_code:
            generated_code = generated_code.split("```html")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()

        try:
            new_app = App(user_id=session['user_id'], prompt=user_prompt, code=generated_code)
            db.session.add(new_app)
            db.session.commit()
        except Exception as db_err:
            db.session.rollback()
            print(f"Failed to save app to history: {db_err}")

        return jsonify({'status': 'success', 'code': generated_code})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Gemini API Error: {str(e)}"}), 500

# --- ENHANCE PROMPT ROUTE ---

@app.route('/enhance-prompt', methods=['POST'])
def enhance_prompt():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Please login first!'}), 401

    data = request.get_json() or {}
    raw_prompt = str(data.get('prompt') or '')
    custom_api_key = str(data.get('api_key') or '').strip()

    if not raw_prompt:
        return jsonify({'status': 'error', 'message': 'Prompt required'}), 400

    active_api_key = custom_api_key if custom_api_key else DEFAULT_GEMINI_API_KEY
    if not active_api_key:
        return jsonify({'status': 'error', 'message': 'Gemini API Key missing!'}), 400

    try:
        enhanced_text = call_gemini_model(f"Expand and detail this web app UI request for high quality generation: {raw_prompt}", api_key=active_api_key)
        return jsonify({'status': 'success', 'enhanced_prompt': enhanced_text.strip()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- AUTO FIX ROUTE ---

@app.route('/auto-fix', methods=['POST'])
def auto_fix():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Please login first!'}), 401

    data = request.get_json() or {}
    code = data.get('code', '')
    error_msg = data.get('error', '')
    custom_api_key = str(data.get('api_key') or '').strip()

    active_api_key = custom_api_key if custom_api_key else DEFAULT_GEMINI_API_KEY
    if not active_api_key:
        return jsonify({'status': 'error', 'message': 'Gemini API Key missing!'}), 400

    try:
        prompt = f"Fix the JavaScript/HTML error in this code.\nError: {error_msg}\nCode:\n{code}\nReturn ONLY updated raw HTML."
        fixed_code = call_gemini_model(prompt, api_key=active_api_key)
        
        if "```html" in fixed_code:
            fixed_code = fixed_code.split("```html")[1].split("```")[0].strip()
        elif "```" in fixed_code:
            fixed_code = fixed_code.split("```")[1].split("```")[0].strip()

        return jsonify({'status': 'success', 'fixed_code': fixed_code})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- HISTORY & DELETE ROUTES ---

@app.route('/history', methods=['GET'])
def history():
    if 'user_id' not in session:
        return jsonify({'history': []})

    user_apps = App.query.filter_by(user_id=session['user_id']).order_by(App.id.desc()).all()
    history_data = [[a.id, a.prompt, a.code] for a in user_apps]
        
    return jsonify({'history': history_data})

@app.route('/history/delete/<int:app_id>', methods=['DELETE'])
@app.route('/delete-app/<int:app_id>', methods=['DELETE'])
def delete_app(app_id):
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

    app_item = App.query.filter_by(id=app_id, user_id=session['user_id']).first()
    if app_item:
        db.session.delete(app_item)
        db.session.commit()
        
    return jsonify({'status': 'success'})

@app.route('/history/clear', methods=['DELETE'])
def clear_history():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

    App.query.filter_by(user_id=session['user_id']).delete()
    db.session.commit()
        
    return jsonify({'status': 'success'})

# --- MAIN SERVER RUNNER ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Server running on port %s", port)
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )