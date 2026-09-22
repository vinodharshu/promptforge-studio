import os
import time
import sqlite3
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "86df73fed8e2613cb2377763562160e7aa2807b6bb6d4cdeaa85dfe1b53c0352")

# 1. Session Configuration for Localhost & Production
app.config['SESSION_COOKIE_SAMESITE'] = 'None'  # Cross-origin cookie handling
app.config['SESSION_COOKIE_SECURE'] = True      # HTTPS requirement

# CORS configuration
CORS(app, supports_credentials=True, origins=[
    "http://127.0.0.1:5500", 
    "http://localhost:5500",
    "http://localhost:5000",
    "https://promptforge-studio.onrender.com"
])

# DEFAULT API KEY
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            api_key TEXT DEFAULT ''
        )
    ''')
    
    # 2. Apps Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS apps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prompt TEXT NOT NULL,
            code TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Automatic Migration check
    cursor.execute("PRAGMA table_info(apps)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'user_id' not in columns:
        cursor.execute("ALTER TABLE apps ADD COLUMN user_id INTEGER")

    conn.commit()
    conn.close()

init_db()

# Helper function to generate content with fallback models
def call_gemini_model(prompt_text):
    models_to_try = [
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'gemini-2.0-flash',
        'gemini-2.5-flash',
        'gemini-3.6-flash'
    ]
    last_exception = None
     
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt_text)
            return response.text
        except Exception as e:
            last_exception = e
            print(f"Model {model_name} failed: {e}. Trying next fallback...")
            
    raise last_exception

# --- USER AUTHENTICATION ROUTES ---

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({'status': 'error', 'message': 'Username and password required!'}), 400

    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': 'User registered successfully!'})
    except sqlite3.IntegrityError:
        return jsonify({'status': 'error', 'message': 'Username already exists!'}), 400

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, username FROM users WHERE username = ? AND password = ?', (username, password))
    user = cursor.fetchone()
    conn.close()

    if user:
        session['user_id'] = user[0]
        session['username'] = user[1]
        return jsonify({'status': 'success', 'username': user[1]})
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

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO apps (user_id, prompt, code) VALUES (?, ?, ?)', 
                       (session['user_id'], user_prompt, generated_code))
        conn.commit()
        conn.close()

        return jsonify({'status': 'success', 'code': generated_code})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Gemini API Error: {str(e)}"}), 500

# --- ENHANCE PROMPT ROUTE ---

@app.route('/enhance-prompt', methods=['POST'])
def enhance_prompt():
    data = request.get_json() or {}
    raw_prompt = data.get('prompt', '')
    
    if not raw_prompt:
        return jsonify({'status': 'error', 'message': 'Prompt required'}), 400

    try:
        genai.configure(api_key=DEFAULT_GEMINI_API_KEY)
        enhanced_text = call_gemini_model(f"Expand and detail this web app UI request for high quality generation: {raw_prompt}")
        return jsonify({'status': 'success', 'enhanced_prompt': enhanced_text.strip()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- AUTO FIX ROUTE ---

@app.route('/auto-fix', methods=['POST'])
def auto_fix():
    data = request.get_json() or {}
    code = data.get('code', '')
    error_msg = data.get('error', '')
    custom_api_key = data.get('api_key', '').strip()

    active_api_key = custom_api_key if custom_api_key else DEFAULT_GEMINI_API_KEY

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

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, prompt, code FROM apps WHERE user_id = ? ORDER BY id DESC', (session['user_id'],))
    rows = cursor.fetchall()
    conn.close()
    return jsonify({'history': rows})

@app.route('/history/delete/<int:app_id>', methods=['DELETE'])
@app.route('/delete-app/<int:app_id>', methods=['DELETE'])
def delete_app(app_id):
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM apps WHERE id = ? AND user_id = ?', (app_id, session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/history/clear', methods=['DELETE'])
def clear_history():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM apps WHERE user_id = ?', (session['user_id'],))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/')
def index():
    return jsonify({"status": "Backend running with Local SQL Database!"})

# --- MAIN SERVER RUNNER ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)