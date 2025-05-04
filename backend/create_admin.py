from db import users_collection
from flask_bcrypt import Bcrypt
from datetime import datetime

bcrypt = Bcrypt()

admin_user = {
    "email": "admin@fluent.com",
    "username": "admin",
    "password": bcrypt.generate_password_hash("admin123").decode('utf-8'),
    "gender": "male",
    "occupation": "System Administrator",
    "is_admin": True,
    "created_at": datetime.now()
}

# Delete if exists
users_collection.delete_one({"username": "admin"})

# Insert new admin
result = users_collection.insert_one(admin_user)
print(f"Admin user created with ID: {result.inserted_id}")