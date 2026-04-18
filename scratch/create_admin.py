import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import SessionLocal
from src import models, auth

def create_initial_admin():
    db = SessionLocal()
    try:
        # Check if admin already exists
        admin = db.query(models.User).filter(models.User.username == "admin").first()
        if admin:
            print("Admin user already exists.")
            return

        hashed_password = auth.get_password_hash("Admin@123")
        new_admin = models.User(
            email="admin@medsarthi.com",
            username="admin",
            hashed_password=hashed_password,
            role=models.UserRole.ADMIN,
            is_active=True
        )
        db.add(new_admin)
        db.commit()
        print("Admin user created successfully!")
        print("Username: admin")
        print("Password: Admin@123")
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_initial_admin()
