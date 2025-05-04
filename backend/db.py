from pymongo import MongoClient
from flask_bcrypt import Bcrypt


# Initialize MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["flutterauth"]

# Define collections
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