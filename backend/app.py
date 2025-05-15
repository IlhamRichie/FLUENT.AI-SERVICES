import os
from flask import Flask, request, jsonify, Blueprint, render_template, send_from_directory, flash, redirect, url_for, session
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from functools import wraps
import cv2
from speech_recognition import Recognizer, AudioFile
import tempfile
import base64
import numpy as np
from datetime import datetime, timedelta
import jwt
from pymongo import MongoClient
from bson.objectid import ObjectId
from db import users_collection, questions_collection, sessions_collection
from flask_swagger_ui import get_swaggerui_blueprint
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/*": {"origins": "*"}
})

# Initialize Bcrypt
bcrypt = Bcrypt(app)

SWAGGER_URL = '/api/docs/swagger'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Fluent Interview API"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET', 'your-very-secret-key-here')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'another-secret-key-for-sessions')
app.config['INACTIVITY_DAYS'] = 3  # Days before user is marked inactive

# MongoDB Configuration
client = MongoClient("mongodb://localhost:27017/")
db = client["flutterauth"]

# Collections
users_collection = db["users"]
questions_collection = db["questions"]
sessions_collection = db["interview_sessions"]

# Initialize sample questions if collection is empty
if questions_collection.count_documents({}) == 0:
    questions_collection.insert_many([
        {
            "question": "Ceritakan tentang diri Anda",
            "category": "general",
            "ideal_answer_keywords": ["pengalaman", "pendidikan", "kemampuan", "tujuan"]
        },
        {
            "question": "Apa kelebihan dan kelemahan Anda?",
            "category": "general",
            "ideal_answer_keywords": ["kelebihan", "kelemahan", "perbaikan", "pengembangan"]
        }
    ])

# Detectors
from detectors.emotion_detector import detect_emotion_status
from detectors.mouth_detector import detect_mouth_status
from detectors.pose_detector import detect_pose_status

def get_user_by_email(email):
    """Utility function to get user by email"""
    return users_collection.find_one({"email": email})

# JWT Token Required Decorator
# Update the token_required decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]  # Bearer <token>
            except IndexError:
                return jsonify({'message': 'Bearer token malformed'}), 401
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
            current_user = users_collection.find_one({"email": data['email']})  # Changed to use email
            
            # Check if user is active
            if not current_user.get('is_active', True):
                return jsonify({'message': 'Your account is inactive'}), 403
                
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid!'}), 401
        except Exception as e:
            return jsonify({'message': str(e)}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

# Admin decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'email' not in session:
            flash('Please login first', 'danger')
            return redirect(url_for('admin_login'))
        
        user = users_collection.find_one({"email": session['email']})
        if not user or not user.get('is_admin', False):
            flash('You do not have admin privileges', 'danger')
            return redirect(url_for('index'))
        
        return f(*args, **kwargs)
    return decorated_function

# Function to check and update inactive users
def check_inactive_users():
    inactive_threshold = datetime.utcnow() - timedelta(days=app.config['INACTIVITY_DAYS'])
    users_collection.update_many(
        {
            "last_login": {"$lt": inactive_threshold},
            "is_active": True
        },
        {"$set": {"is_active": False}}
    )

# Create blueprint for interview routes
interview_blueprint = Blueprint('interview', __name__)

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    required_fields = ["email", "username", "password", "gender", "occupation"]
    if not all(field in data for field in required_fields):
        return jsonify({"status": "fail", "message": "Incomplete data"}), 400

    # Check if username or email exists
    if users_collection.find_one({"$or": [{"username": data["username"]}, {"email": data["email"]}]}):
        return jsonify({"status": "fail", "message": "Username or email already exists"}), 400

    hashed_password = bcrypt.generate_password_hash(data["password"]).decode('utf-8')
    
    user_data = {
        "email": data["email"],
        "username": data["username"],
        "password": hashed_password,
        "gender": data["gender"],
        "occupation": data["occupation"],
        "is_active": True,
        "last_login": None,
        "created_at": datetime.utcnow(),
        "is_admin": data.get("is_admin", False)  # Add this field
    }
    
    try:
        user_id = users_collection.insert_one(user_data).inserted_id
        return jsonify({
            "status": "success",
            "message": "User registered successfully",
            "user": {
                "id": str(user_id),
                "username": data["username"],
                "email": data["email"]
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"status": "fail", "message": "Missing email or password"}), 400

    user = users_collection.find_one({"email": data["email"]})
    if not user:
        return jsonify({"status": "fail", "message": "User not found"}), 404

    # Check if account is active
    if not user.get('is_active', True):
        return jsonify({"status": "fail", "message": "Account is inactive"}), 403

    if not bcrypt.check_password_hash(user['password'], data['password']):
        return jsonify({"status": "fail", "message": "Invalid password"}), 401
    
    # Update last login
    users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.utcnow(), "is_active": True}}
    )
    
    # Generate tokens
    access_token = jwt.encode({
        'email': user['email'],  # Changed from username to email
        'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
    }, app.config['JWT_SECRET_KEY'], algorithm="HS256")

    refresh_token = jwt.encode({
        'email': user['email'],  # Changed from username to email
        'exp': datetime.utcnow() + app.config['JWT_REFRESH_TOKEN_EXPIRES']
    }, app.config['JWT_SECRET_KEY'], algorithm="HS256")

    return jsonify({
        "status": "success",
        "message": "Login successful",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": {
            "username": user["username"],
            "email": user["email"],
            "gender": user["gender"],
            "occupation": user["occupation"],
            "is_active": True
        }
    })

@app.route("/refresh", methods=["POST"])
def refresh():
    refresh_token = request.json.get('refresh_token')
    if not refresh_token:
        return jsonify({"status": "fail", "message": "Refresh token is missing"}), 401
    
    try:
        data = jwt.decode(refresh_token, app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
        user = get_user_by_email(data['email'])  # Changed to use email
        
        if not user:
            return jsonify({"status": "fail", "message": "User not found"}), 404
        
        new_access_token = jwt.encode({
            'email': user['email'],  # Changed from username to email
            'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
        }, app.config['JWT_SECRET_KEY'], algorithm="HS256")
        
        return jsonify({
            "status": "success",
            "access_token": new_access_token
        })
    except jwt.ExpiredSignatureError:
        return jsonify({"status": "fail", "message": "Refresh token has expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"status": "fail", "message": "Invalid refresh token"}), 401

@app.route("/analyze_realtime", methods=["POST"])
@token_required
def analyze_realtime(current_user):
    data = request.get_json()

    if "frame" not in data:
        return jsonify({"status": "fail", "message": "Frame not provided"}), 400

    try:
        # Decode base64 to numpy array
        img_data = base64.b64decode(data["frame"])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "fail", "message": "Invalid image data"}), 400

        # Save temporary image for processing
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            image_path = tmp_file.name
            cv2.imwrite(image_path, frame)

        # Run all detectors
        emotion_result = detect_emotion_status(image_path)
        mouth_result = detect_mouth_status(image_path)
        pose_result = detect_pose_status(image_path)

        # Clean up
        os.unlink(image_path)

        return jsonify({
            "status": "success",
            "results": {
                "emotion": emotion_result,
                "mouth": mouth_result,
                "pose": pose_result
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
    
@app.route('/user/update', methods=['PUT'])
@token_required
def update_user(current_user):
    data = request.get_json()
    update_fields = {}

    if "username" in data:
        update_fields["username"] = data["username"]
    if "occupation" in data:
        update_fields["occupation"] = data["occupation"]

    if not update_fields:
        return jsonify({"status": "fail", "message": "No valid fields to update"}), 400

    users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": update_fields}
    )

    return jsonify({"status": "success", "message": "User data updated"})

    
@app.route("/analyze_speech", methods=["POST"])
@token_required
def analyze_speech(current_user):
    if 'audio' not in request.files:
        return jsonify({"status": "fail", "message": "No audio file"}), 400
    
    audio_file = request.files['audio']
    
    # Save to temp file
    temp_path = os.path.join(tempfile.gettempdir(), "interview_audio.wav")
    audio_file.save(temp_path)
    
    try:
        recognizer = Recognizer()
        with AudioFile(temp_path) as source:
            audio = recognizer.record(source)
        
        # Perform speech recognition
        text = recognizer.recognize_google(audio, language="id-ID")
        
        # Basic analysis
        words = text.split()
        wpm = len(words) / (30 / 60)  # Assuming 30 second recording
        
        return jsonify({
            "status": "success",
            "transcript": text,
            "word_count": len(words),
            "words_per_minute": wpm,
            "language": "id-ID"
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Interview Routes
@interview_blueprint.route('/start', methods=['POST'])
@token_required
def start_interview(current_user):
    try:
        data = request.get_json()
        category = data.get('category', 'general')
        
        # Get random questions
        questions = list(questions_collection.aggregate([
            {"$match": {"category": category}},
            {"$sample": {"size": 5}}
        ]))
        
        if not questions:
            return jsonify({"status": "fail", "message": "No questions available"}), 404
        
        # Create session
        session_data = {
            "user_id": current_user["_id"],
            "start_time": datetime.now(),
            "status": "ongoing",
            "questions": [{
                "question_id": str(q["_id"]),
                "question_text": q["question"],
                "ideal_keywords": q.get("ideal_answer_keywords", []),
                "user_answer": None,
                "evaluation": None
            } for q in questions],
            "current_question_index": 0,
            "category": category
        }
        
        session_id = sessions_collection.insert_one(session_data).inserted_id
        
        return jsonify({
            "status": "success",
            "session_id": str(session_id),
            "current_question": session_data["questions"][0]["question_text"],
            "current_question_id": session_data["questions"][0]["question_id"],
            "total_questions": len(session_data["questions"])
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@interview_blueprint.route('/submit', methods=['POST'])
@token_required
def submit_answer(current_user):
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        answer_text = data.get("answer_text")
        audio_answer = data.get("audio_answer", None)
        
        if not all([session_id, answer_text]):
            return jsonify({"status": "fail", "message": "Missing data"}), 400
        
        # Get session from database
        session = sessions_collection.find_one({"_id": session_id, "user_id": current_user["_id"]})
        if not session:
            return jsonify({"status": "fail", "message": "Session not found"}), 404
        
        current_idx = session["current_question_index"]
        if current_idx >= len(session["questions"]):
            return jsonify({"status": "fail", "message": "Interview completed"}), 400
        
        # Evaluate answer (simple keyword matching)
        question = session["questions"][current_idx]
        matched_keywords = [kw for kw in question["ideal_keywords"] 
                          if kw.lower() in answer_text.lower()]
        score = len(matched_keywords) / len(question["ideal_keywords"]) * 100 if question["ideal_keywords"] else 0
        
        evaluation = {
            "matched_keywords": matched_keywords,
            "score": round(score, 2),
            "feedback": "Good answer" if score > 50 else "Needs improvement"
        }
        
        # Update session
        update_data = {
            f"questions.{current_idx}.user_answer": answer_text,
            f"questions.{current_idx}.evaluation": evaluation,
            "current_question_index": current_idx + 1
        }
        
        if current_idx + 1 >= len(session["questions"]):
            update_data["status"] = "completed"
            update_data["end_time"] = datetime.now()
            completed = True
        else:
            completed = False
        
        sessions_collection.update_one(
            {"_id": session_id},
            {"$set": update_data}
        )
        
        response = {
            "status": "success",
            "evaluation": evaluation,
            "interview_completed": completed
        }
        
        if completed:
            response["overall_score"] = sum(
                q.get("evaluation", {}).get("score", 0) 
                for q in session["questions"]
            ) / len(session["questions"])
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@interview_blueprint.route('/results/<session_id>', methods=['GET'])
@token_required
def get_results(current_user, session_id):
    try:
        session = sessions_collection.find_one({"_id": session_id, "user_id": current_user["_id"]})
        if not session or session["status"] != "completed":
            return jsonify({"status": "fail", "message": "Results not available"}), 404
        
        return jsonify({
            "status": "success",
            "results": {
                "session_id": str(session["_id"]),
                "user_id": session["user_id"],
                "start_time": session["start_time"],
                "end_time": session.get("end_time"),
                "questions": session["questions"],
                "overall_score": sum(
                    q.get("evaluation", {}).get("score", 0) 
                    for q in session["questions"]
                ) / len(session["questions"])
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Admin routes
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = get_user_by_email(email)
        if user and user.get('is_admin', False) and bcrypt.check_password_hash(user['password'], password):
            session['email'] = user['email']
            session['is_admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid credentials or not an admin', 'danger')
    
    return render_template('admin_login.html')

@app.route('/api/docs')
def api_docs():
    """API Documentation Page"""
    return render_template('api_docs.html')

@app.route('/admin/sessions')
@admin_required
def admin_sessions():
    """View all interview sessions"""
    sessions = list(sessions_collection.find().sort("start_time", -1))
    return render_template('admin_sessions.html', sessions=sessions)

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    check_inactive_users()

    # Ambil data pengguna
    user_count = users_collection.count_documents({})
    today_sessions = sessions_collection.count_documents({
        "start_time": {
            "$gte": datetime.combine(datetime.today(), datetime.min.time())
        }
    })

    # Data registrasi pengguna
    pipeline = [
        {"$project": {
            "year": {"$year": "$created_at"},
            "month": {"$month": "$created_at"},
            "day": {"$dayOfMonth": "$created_at"}
        }},
        {"$group": {
            "_id": {"year": "$year", "month": "$month", "day": "$day"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}}
    ]
    reg_data = list(users_collection.aggregate(pipeline))
    df_reg = pd.DataFrame(reg_data)
    if not df_reg.empty:
        df_reg['date'] = pd.to_datetime(df_reg['_id'].apply(lambda x: f"{x['year']}-{x['month']}-{x['day']}"))
        fig_registration = px.line(df_reg, x='date', y='count', title='User Registration Trend')
        registration_chart = fig_registration.to_html(full_html=False)
    else:
        registration_chart = "<p>No registration data</p>"

    # Gender Distribution
    gender_counts = users_collection.aggregate([
        {"$group": {"_id": "$gender", "count": {"$sum": 1}}}
    ])
    df_gender = pd.DataFrame(gender_counts)
    if not df_gender.empty:
        fig_gender = px.pie(df_gender, names='_id', values='count', title='Gender Distribution')
        gender_chart = fig_gender.to_html(full_html=False)
    else:
        gender_chart = "<p>No gender data</p>"

    # Occupation Distribution
    occ_counts = users_collection.aggregate([
        {"$group": {"_id": "$occupation", "count": {"$sum": 1}}}
    ])
    df_occ = pd.DataFrame(occ_counts)
    if not df_occ.empty:
        fig_occ = px.pie(df_occ, names='_id', values='count', title='Occupation Distribution')
        occupation_chart = fig_occ.to_html(full_html=False)
    else:
        occupation_chart = "<p>No occupation data</p>"

    # User Activity Overview
    session_pipeline = [
        {"$project": {
            "year": {"$year": "$start_time"},
            "month": {"$month": "$start_time"},
            "day": {"$dayOfMonth": "$start_time"}
        }},
        {"$group": {
            "_id": {"year": "$year", "month": "$month", "day": "$day"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}}
    ]
    act_data = list(sessions_collection.aggregate(session_pipeline))
    df_act = pd.DataFrame(act_data)
    if not df_act.empty:
        df_act['date'] = pd.to_datetime(df_act['_id'].apply(lambda x: f"{x['year']}-{x['month']}-{x['day']}"))
        fig_activity = px.bar(df_act, x='date', y='count', title='Daily Interview Sessions')
        activity_chart = fig_activity.to_html(full_html=False)
    else:
        activity_chart = "<p>No activity data</p>"

    # Data lainnya
    recent_activities = list(db.activity_log.find().sort("time", -1).limit(10)) if hasattr(db, 'activity_log') else []
    all_users = list(users_collection.find({}, {
        "_id": 1,
        "username": 1,
        "email": 1,
        "gender": 1,
        "occupation": 1,
        "created_at": 1,
        "is_active": 1,
        "last_login": 1
    }).sort("created_at", -1))

    return render_template('admin_dashboard.html',
                           user_count=user_count,
                           today_sessions=today_sessions,
                           recent_activities=recent_activities,
                           all_users=all_users,
                           registration_chart=registration_chart,
                           gender_chart=gender_chart,
                           occupation_chart=occupation_chart,
                           activity_chart=activity_chart)
    
@app.route('/admin/users/toggle-status/<user_id>', methods=['POST'])
@admin_required
def toggle_user_status(user_id):
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            flash('User not found', 'danger')
            return redirect(url_for('admin_dashboard'))
        
        new_status = not user.get('is_active', False)
        users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"is_active": new_status}}
        )
        
        flash(f'User status changed to {"Active" if new_status else "Inactive"}', 'success')
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/logout')
def admin_logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('admin_login'))

# Register the blueprint
app.register_blueprint(interview_blueprint, url_prefix='/api/interview')

# Website routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    # Run the inactivity check when starting the app
    with app.app_context():
        check_inactive_users()
    app.run(host="0.0.0.0", port=5000, debug=True)
