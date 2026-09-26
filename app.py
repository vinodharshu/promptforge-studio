# app.py
import os, logging
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from datetime import date
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import SQLAlchemyError
from importlib import import_module

# Use new Google GenAI SDK (deprecated google-generativeai)
from google import genai  

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__, static_folder='.', template_folder='.')
# 1. SECRET_KEY (required)
configured_secret = os.getenv("SECRET_KEY", "").strip()
if not configured_secret:
    logger.error("SECRET_KEY is not set in environment!")
    raise RuntimeError("SECRET_KEY environment variable must be set")
app.secret_key = configured_secret
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = True  # assume production (HTTPS)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# 2. Database configuration (Postgres recommended)
db_url = os.getenv("DATABASE_URL", "").strip()
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
if not db_url:
    logger.warning("DATABASE_URL not set; using SQLite (not persistent)")
    db_url = "sqlite:///database.db"
elif db_url.startswith("sqlite"):
    logger.warning("Using SQLite on Render is not persistent! Use Postgres.")

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
SQLAlchemy = import_module('flask_sqlalchemy').SQLAlchemy
db = SQLAlchemy(app)

# 3. CORS (allow frontend origin if needed; here wildcard for same origin + Render)
CORS(app, supports_credentials=True, origins=["*"])

# Models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    # per-user daily usage tracking
    generation_count = db.Column(db.Integer, default=0)
    enhancement_count = db.Column(db.Integer, default=0)
    autofix_count = db.Column(db.Integer, default=0)
    usage_date = db.Column(db.Date, nullable=True)
    apps = db.relationship('App', backref='owner', lazy=True)

class App(db.Model):
    __tablename__ = 'apps'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    code = db.Column(db.Text, nullable=False)

# Initialize DB
try:
    with app.app_context():
        db.create_all()
    logger.info("Database initialization completed.")
except Exception as e:
    logger.exception("Database initialization failed.")

# 4. Gemini API Key (required)
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip("'").strip('"')
if not DEFAULT_GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not set in environment!")
    raise RuntimeError("GEMINI_API_KEY environment variable must be set")

# Helper: reset usage counts if a new day
def reset_usage_if_needed(user):
    today = date.today()
    if user.usage_date != today:
        user.generation_count = 0
        user.enhancement_count = 0
        user.autofix_count = 0
        user.usage_date = today
        db.session.commit()

# Helper: enforce per-user quotas (e.g. 10 operations/day)
DAILY_LIMIT = 10
def increment_usage(user, field):
    reset_usage_if_needed(user)
    if getattr(user, field) >= DAILY_LIMIT:
        return False
    setattr(user, field, getattr(user, field) + 1)
    user.usage_date = date.today()
    db.session.commit()
    return True

# Gemini API call (GenAI SDK)
def call_gemini_model(prompt_text):
    client = genai.Client(api_key=DEFAULT_GEMINI_API_KEY)
    try:
        # Generate content with system instruction handled outside
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt_text
        )
        return response.text or ""
    except Exception as e:
        err = str(e).lower()
        if "exhausted" in err or "quota" in err or "429" in err:
            raise RuntimeError("QuotaExceeded") from e
        raise

@app.errorhandler(Exception)
def handle_exception(e):
    if hasattr(e, 'code'):
        return jsonify({'status':'error','error_type':'server','message': e.description}), e.code
    logger.exception("Unhandled exception")
    return jsonify({'status':'error','error_type':'server','message':'Internal server error'}), 500

# --- Routes ---

@app.route('/')
def index():
    # Serve the index.html (static in root)
    return send_from_directory('.', 'index.html')

@app.route('/api/health')
def health():
    return jsonify({
        "status": "ok",
        "database_configured": bool(os.getenv("DATABASE_URL")),
        "gemini_configured": bool(DEFAULT_GEMINI_API_KEY)
    }), 200

# Authentication
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    username = data.get('username','').strip()
    password = data.get('password','').strip()
    if not username or not password:
        return jsonify({'status':'error','error_type':'auth','message':'Username and password required!'}), 400
    try:
        if User.query.filter_by(username=username).first():
            return jsonify({'status':'error','error_type':'auth','message':'Username already exists!'}), 400
        hashed = generate_password_hash(password)
        user = User(username=username, password=hashed)
        db.session.add(user); db.session.commit()
        return jsonify({'status':'success','message':'User registered successfully!'})
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"DB error on register: {e}")
        return jsonify({'status':'error','error_type':'db','message':'Database error. Try again later.'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username','').strip()
    password = data.get('password','').strip()
    try:
        user = User.query.filter_by(username=username).first()
    except SQLAlchemyError as e:
        logger.error(f"DB error on login: {e}")
        return jsonify({'status':'error','error_type':'db','message':'Database error. Try again later.'}), 500
    if user and check_password_hash(user.password, password):
        session['user_id'] = user.id
        session['username'] = user.username
        session.modified = True
        return jsonify({'status':'success','username': user.username})
    return jsonify({'status':'error','error_type':'auth','message':'Invalid credentials!'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'status':'success','message':'Logged out'})

@app.route('/api/user-status', methods=['GET'])
def user_status():
    if 'user_id' in session:
        return jsonify({'logged_in': True, 'username': session['username']})
    return jsonify({'logged_in': False})

# --- Generate App ---
@app.route('/api/generate', methods=['POST'])
def generate():
    if 'user_id' not in session:
        return jsonify({'status':'error','error_type':'auth','message':'Please login first!'}), 401
    data = request.get_json() or {}
    user_prompt = str(data.get('prompt',''))
    prev_code = data.get('previous_code','')
    if not user_prompt:
        return jsonify({'status':'error','error_type':'request','message':'Prompt is required!'}), 400

    # Enforce per-user quota
    user = User.query.get(session['user_id'])
    reset_usage_if_needed(user)
    if not increment_usage(user, 'generation_count'):
        return jsonify({'status':'error','error_type':'quota','message':'Daily generation quota reached'}), 429

    # System instruction for generation
    system_instruction = (
        "You are a world-class UI/UX and web developer. "
        "Generate a single-file, mobile-responsive HTML+CSS (Tailwind) web app based on the user prompt. "
        "Use Tailwind CSS breakpoints (sm, md, lg, etc.) for mobile-first design, and ensure no horizontal scrolling. "
        "Output only valid HTML (with <!DOCTYPE html>), no markdown formatting or code fences. "
        "If updating existing code, preserve the core app and only apply requested changes."
    )
    full_prompt = f"{system_instruction}\n\nUSER PROMPT: {user_prompt}\n"
    if prev_code:
        full_prompt += f"\nPREVIOUS CODE TO MODIFY:\n{prev_code}"

    try:
        generated_code = call_gemini_model(full_prompt)
    except RuntimeError as e:
        # Quota exceeded by Gemini
        return jsonify({'status':'error','error_type':'quota','message':'Gemini API quota exceeded'}), 429
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return jsonify({'status':'error','error_type':'gemini','message':'AI generation failed'}), 500

    # Strip Markdown fences if present
    if "```" in generated_code:
        parts = generated_code.split("```")
        generated_code = parts[1] if len(parts)>2 else parts[0]

    # Basic validation
    if "<!DOCTYPE html" not in generated_code[:50]:
        logger.error("Invalid HTML output from Gemini")
        return jsonify({'status':'error','error_type':'format','message':'Invalid HTML output from AI'}), 502

    # Save to history (ignore DB errors)
    try:
        new_app = App(user_id=session['user_id'], prompt=user_prompt, code=generated_code)
        db.session.add(new_app)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.warning(f"Failed to save history: {e}")

    return jsonify({'status':'success','code': generated_code})

# --- Enhance Prompt ---
@app.route('/api/enhance-prompt', methods=['POST'])
def enhance_prompt():
    if 'user_id' not in session:
        return jsonify({'status':'error','error_type':'auth','message':'Please login first!'}), 401
    data = request.get_json() or {}
    raw_prompt = str(data.get('prompt','')).strip()
    if not raw_prompt:
        return jsonify({'status':'error','error_type':'request','message':'Prompt required'}), 400

    # Enforce per-user quota
    user = User.query.get(session['user_id'])
    reset_usage_if_needed(user)
    if not increment_usage(user, 'enhancement_count'):
        return jsonify({'status':'error','error_type':'quota','message':'Daily enhancement quota reached'}), 429

    try:
        enhanced_text = call_gemini_model(f"Improve this UI request for clarity and detail: {raw_prompt}")
        return jsonify({'status':'success','enhanced_prompt': enhanced_text.strip()})
    except RuntimeError as e:
        return jsonify({'status':'error','error_type':'quota','message':'Gemini API quota exceeded'}), 429
    except Exception as e:
        logger.error(f"Enhance prompt error: {e}")
        return jsonify({'status':'error','error_type':'gemini','message': str(e)}), 500

# --- Auto-Fix JS/HTML Error ---
@app.route('/api/auto-fix', methods=['POST'])
def auto_fix():
    if 'user_id' not in session:
        return jsonify({'status':'error','error_type':'auth','message':'Please login first!'}), 401
    data = request.get_json() or {}
    code = data.get('code','')
    error_msg = data.get('error','')
    if not code or not error_msg:
        return jsonify({'status':'error','error_type':'request','message':'Code and error required'}), 400

    # Enforce per-user quota
    user = User.query.get(session['user_id'])
    reset_usage_if_needed(user)
    if not increment_usage(user, 'autofix_count'):
        return jsonify({'status':'error','error_type':'quota','message':'Daily auto-fix quota reached'}), 429

    prompt = f"Fix the JavaScript/HTML error in this code.\nError: {error_msg}\nCode:\n{code}\nReturn only the corrected HTML code without explanation."
    try:
        fixed_code = call_gemini_model(prompt)
    except RuntimeError:
        return jsonify({'status':'error','error_type':'quota','message':'Gemini API quota exceeded'}), 429
    except Exception as e:
        logger.error(f"Auto-fix error: {e}")
        return jsonify({'status':'error','error_type':'gemini','message': str(e)}), 500

    if "```" in fixed_code:
        parts = fixed_code.split("```")
        fixed_code = parts[1] if len(parts)>2 else parts[0]
    return jsonify({'status':'success','fixed_code': fixed_code})

# --- History & Deletion ---
@app.route('/api/history', methods=['GET'])
def history():
    if 'user_id' not in session:
        return jsonify({'history': []})
    user_apps = App.query.filter_by(user_id=session['user_id']).order_by(App.id.desc()).all()
    history_data = [[a.id, a.prompt, a.code] for a in user_apps]
    return jsonify({'history': history_data})

@app.route('/api/history/delete/<int:app_id>', methods=['DELETE'])
def delete_app(app_id):
    if 'user_id' not in session:
        return jsonify({'status':'error','error_type':'auth','message':'Unauthorized'}), 401
    app_item = App.query.filter_by(id=app_id, user_id=session['user_id']).first()
    if app_item:
        db.session.delete(app_item); db.session.commit()
    return jsonify({'status':'success'})

@app.route('/api/history/clear', methods=['DELETE'])
def clear_history():
    if 'user_id' not in session:
        return jsonify({'status':'error','error_type':'auth','message':'Unauthorized'}), 401
    App.query.filter_by(user_id=session['user_id']).delete()
    db.session.commit()
    return jsonify({'status':'success'})

# --- Run Server ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Server starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
