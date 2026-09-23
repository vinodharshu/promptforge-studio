import os
import secrets
from flask import Flask, send_from_directory, request, jsonify, session
from flask_cors import CORS
from importlib import import_module
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai

# Load the optional extension dynamically so static analyzers do not report a
# missing direct import when the dependency is installed only at runtime.
SQLAlchemy = import_module('flask_sqlalchemy').SQLAlchemy

app = Flask(__name__, static_folder='.', template_folder='.')

# Are we running locally (dev) or deployed (prod)?
# Render (and most PaaS hosts) set PORT; treat FLASK_ENV=development as the
# explicit local-dev override.
IS_PROD = os.getenv("FLASK_ENV", "production").lower() != "development"

# 1. Secret Key setup for sessions
# NEVER hardcode a real secret key in source control — a previous version of
# this file shipped a fixed fallback key, which means every deployment that
# didn't set SECRET_KEY shared (and leaked) the same signing key, letting
# anyone forge session cookies. Always set SECRET_KEY in your environment for
# production; the random fallback below only keeps local/dev runs working and
# will invalidate sessions on every restart.
app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)
if IS_PROD and not os.getenv("SECRET_KEY","e63b91911951c497ab7b17f46bab4a168f8091ab206900e03c477c74f9f166d1"):
    print("⚠️  WARNING: SECRET_KEY is not set. Using a random key that changes "
          "on every restart (all existing sessions will be invalidated). Set "
          "the SECRET_KEY environment variable in production.")

# 2. Session Cookie Handling
# SameSite=None + Secure is required for the cross-origin (different
# subdomain) frontend/backend setup this app uses, but Secure cookies are
# refused by browsers over plain http, which breaks cookie-based login when
# testing locally at http://127.0.0.1:5000. Relax this only in explicit dev.
app.config['SESSION_COOKIE_SAMESITE'] = 'None' if IS_PROD else 'Lax'
app.config['SESSION_COOKIE_SECURE'] = IS_PROD
app.config['SESSION_COOKIE_HTTPONLY'] = True

# 3. SQL DATABASE CONFIGURATION (Supports PostgreSQL, MySQL, SQLite)
# Render's managed Postgres gives a DATABASE_URL starting with "postgres://",
# which SQLAlchemy 1.4+ rejects (it requires "postgresql://"). If DATABASE_URL
# isn't set at all (e.g. no DB attached yet), fall back to a local SQLite file
# instead of crashing — note that Render's filesystem is ephemeral, so SQLite
# data will NOT survive a redeploy/restart there; attach a real Postgres
# instance for anything you need to persist.
db_url = os.getenv("DATABASE_URL", "sqlite:///database.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
if db_url.startswith("postgresql://"):
    # Render's free-tier Postgres silently drops idle connections, which
    # otherwise surfaces as random "SSL connection has been closed
    # unexpectedly" runtime errors on the first request after a quiet period.
    # pool_pre_ping checks the connection before using it and transparently
    # reconnects; pool_recycle keeps connections from going stale.
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'pool_recycle': 280,
    }

db = SQLAlchemy(app)

# 4. CORS Configuration
ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "https://promptforge-studio-frontend.onrender.com"
]
# Let an extra frontend origin (e.g. a preview deploy) be added without a
# code change/redeploy.
extra_origin = os.getenv("https://promptforge-studio-frondend.onrender.com")
if extra_origin and extra_origin not in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS.append(extra_origin)

CORS(app, supports_credentials=True, origins=ALLOWED_ORIGINS)

@app.before_request
def _log_cross_origin_requests():
    # Diagnostic only — helps you see in Render's logs exactly which Origin
    # a failing request came from, so you can confirm it's actually in
    # ALLOWED_ORIGINS. A mismatch here (e.g. your frontend's real Render URL
    # isn't in the list) makes the browser block the response client-side,
    # which shows up in the UI as a generic "Network Error" even though this
    # log line will show the request arrived fine.
    origin = request.headers.get('Origin')
    if origin and origin not in ALLOWED_ORIGINS:
        print(f"⚠️  Request from origin '{origin}' is NOT in ALLOWED_ORIGINS {ALLOWED_ORIGINS} "
              f"— the browser will block this response. Add it via the FRONTEND_URL env var.")

@app.errorhandler(Exception)
def _handle_uncaught_exception(e):
    # Without this, any unhandled exception in a route becomes Flask's
    # default HTML error page. The frontend always does `await res.json()`,
    # and parsing HTML as JSON throws — which lands in the same catch block
    # as an actual network failure and shows "Network Error. Check backend
    # connection.", even though the backend was up and responding. Returning
    # JSON here means real backend errors show up as real backend errors.
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return jsonify({'status': 'error', 'message': e.description}), e.code
    print(f"Unhandled exception: {e}")
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

try:
    with app.app_context():
        db.create_all()
except Exception as db_init_err:
    # If the DB is unreachable at boot (e.g. Postgres not provisioned yet,
    # wrong DATABASE_URL, network hiccup on Render's side), don't take the
    # whole server down — log it loudly and keep serving. Routes that need
    # the DB will fail individually with a clear error instead of the
    # process crash-looping.
    print(f"⚠️  WARNING: Database initialization failed: {db_init_err}")
    print("   The server will still start, but DB-dependent routes "
          "(/api/register, /api/login, /generate, /history, ...) will "
          "error until this is fixed.")

# Helper function to generate content with fallback models
#
# NOTE ON MODEL NAMES: gemini-1.5-flash / gemini-1.5-pro have been fully shut
# down (every request now 404s), and the gemini-2.0-flash line was shut down
# June 1, 2026 — so the previous fallback list would fail on every model.
# "-latest" aliases are used first so Google's automatic version bumps (e.g.
# 2.5 -> 3.x) don't silently break this app again; pinned versions follow as
# a safety net in case an alias is ever retired.
def call_gemini_model(prompt_text):
    models_to_try = [
        'gemini-flash-latest',
        'gemini-3.5-flash',
        'gemini-3.1-flash-lite',
        'gemini-pro-latest',
    ]
    last_exception = Exception("No models responded successfully.")
     
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt_text)
            return response.text
        except Exception as e:
            last_exception = e
            print(f"Model {model_name} failed: {e}. Trying next...")
            
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
        # Previously the query above wasn't wrapped, so a DB outage raised an
        # unhandled exception -> Flask's default HTML 500 page -> the
        # frontend's `await res.json()` throws a SyntaxError trying to parse
        # HTML as JSON -> shows up to the user as "Network Error. Check
        # backend connection." even though the backend was actually up.
        # Always returning JSON here fixes that class of false alarm.
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

# --- HEALTH CHECK (no DB, no Gemini — use this to confirm the backend
# itself is reachable, independent of everything else) ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})



# --- GENERATE APP ROUTE ---

@app.route('/generate', methods=['POST'])
def generate():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Please login first!'}), 401

    data = request.get_json() or {}
    user_prompt = data.get('prompt', '')
    previous_code = data.get('previous_code', '')
    custom_api_key = data.get('api_key', '').strip()

    if not user_prompt:
        return jsonify({'status': 'error', 'message': 'Prompt is required!'}), 400

    active_api_key = custom_api_key if custom_api_key else DEFAULT_GEMINI_API_KEY

    if not active_api_key:
        return jsonify({'status': 'error', 'message': 'Gemini API Key missing!'}), 400

    try:
        genai.configure(api_key=active_api_key)

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

        generated_code = call_gemini_model(full_prompt)

        if "```html" in generated_code:
            generated_code = generated_code.split("```html")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()

        # Save to history, but don't fail the whole request if only the save
        # fails — the user still gets their generated app back either way.
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
    raw_prompt = data.get('prompt', '')
    custom_api_key = data.get('api_key', '').strip()

    if not raw_prompt:
        return jsonify({'status': 'error', 'message': 'Prompt required'}), 400

    active_api_key = custom_api_key if custom_api_key else DEFAULT_GEMINI_API_KEY
    if not active_api_key:
        return jsonify({'status': 'error', 'message': 'Gemini API Key missing!'}), 400

    try:
        genai.configure(api_key=active_api_key)
        enhanced_text = call_gemini_model(f"Expand and detail this web app UI request for high quality generation: {raw_prompt}")
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
    custom_api_key = data.get('api_key', '').strip()

    active_api_key = custom_api_key if custom_api_key else DEFAULT_GEMINI_API_KEY
    if not active_api_key:
        return jsonify({'status': 'error', 'message': 'Gemini API Key missing!'}), 400

    try:
        genai.configure(api_key=active_api_key)
        prompt = f"Fix the JavaScript/HTML error in this code.\nError: {error_msg}\nCode:\n{code}\nReturn ONLY updated raw HTML."
        fixed_code = call_gemini_model(prompt)
        
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
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # debug=True exposes Werkzeug's interactive debugger, which allows remote
    # code execution if it's ever reachable in production. Only enable it
    # when FLASK_ENV=development is explicitly set.
    print(f"🚀 Server running on port {port} (debug={not IS_PROD})")
    app.run(host='0.0.0.0', port=port, debug=not IS_PROD)