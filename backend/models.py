from datetime import datetime
from db import users_collection
from flask_bcrypt import Bcrypt
from bson.objectid import ObjectId

bcrypt = Bcrypt()

def register_user(email, username, password, gender, occupation, is_admin=False):
    """Register a new user with optional admin flag"""
    if users_collection.find_one({"username": username}):
        return {"status": "fail", "message": "Username already exists"}
    if users_collection.find_one({"email": email}):
        return {"status": "fail", "message": "Email already registered"}

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user_data = {
        "email": email,
        "username": username,
        "password": hashed_password,
        "gender": gender,
        "occupation": occupation,
        "is_admin": is_admin,
        "api_keys": [],
        "created_at": datetime.now()
    }
    result = users_collection.insert_one(user_data)
    return {
        "status": "success",
        "message": "User registered",
        "user_id": str(result.inserted_id)
    }

def login_user(username, password):
    """Authenticate a user"""
    user = users_collection.find_one({"username": username})
    if user and bcrypt.check_password_hash(user["password"], password):
        return {"status": "success", "message": "Login successful", "user": user}
    return {"status": "fail", "message": "Invalid credentials"}

def get_user_by_username(username):
    """Get user by username"""
    user = users_collection.find_one({"username": username})
    if user:
        user["_id"] = str(user["_id"])  # Convert ObjectId to string
    return user

def get_admin_users():
    """Get all admin users"""
    return list(users_collection.find({"is_admin": True}))

def verify_api_key(user_id, api_key):
    """Verify if API key is valid for user"""
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        return False
    return api_key in user.get("api_keys", [])