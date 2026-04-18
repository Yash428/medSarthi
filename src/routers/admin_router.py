from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from src import models, schemas, auth
from src.database import get_db
from src.dependencies import get_current_admin

router = APIRouter(prefix="/api/admin", tags=["admin"])

@router.get("/users", response_model=List[schemas.UserResponse])
def get_all_users(db: Session = Depends(get_db), current_admin: models.User = Depends(get_current_admin)):
    """List all users in the system."""
    return db.query(models.User).all()

@router.post("/users", response_model=schemas.UserResponse)
def admin_create_user(user: schemas.UserAdminCreate, db: Session = Depends(get_db), current_admin: models.User = Depends(get_current_admin)):
    """Create a new user with any role."""
    # Check if exists
    db_user = db.query(models.User).filter((models.User.email == user.email) | (models.User.username == user.username)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email or Username already registered")
    
    hashed_password = auth.get_password_hash(user.password)
    new_user = models.User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        role=user.role,
        is_active=user.is_active
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Base profile creation
    if user.role == models.UserRole.PATIENT:
        profile = models.PatientProfile(user_id=new_user.id)
        db.add(profile)
    elif user.role == models.UserRole.DOCTOR:
        profile = models.DoctorProfile(user_id=new_user.id, specialization="General Physician")
        db.add(profile)
    
    db.commit()
    return new_user

@router.put("/users/{user_id}", response_model=schemas.UserResponse)
def admin_update_user(user_id: int, user_data: schemas.UserAdminUpdate, db: Session = Depends(get_db), current_admin: models.User = Depends(get_current_admin)):
    """Update user details."""
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user_data.email:
        db_user.email = user_data.email
    if user_data.username:
        db_user.username = user_data.username
    if user_data.password:
        db_user.hashed_password = auth.get_password_hash(user_data.password)
    if user_data.role is not None:
        db_user.role = user_data.role
    if user_data.is_active is not None:
        db_user.is_active = user_data.is_active
    
    db.commit()
    db.refresh(db_user)
    return db_user

@router.delete("/users/{user_id}")
def admin_delete_user(user_id: int, db: Session = Depends(get_db), current_admin: models.User = Depends(get_current_admin)):
    """Permanently delete a user."""
    if user_id == current_admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
        
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Cascade delete is handled by relationships if configured, but we do explicitly here for safety with profiles
    if db_user.patient_profile:
        db.delete(db_user.patient_profile)
    if db_user.doctor_profile:
        db.delete(db_user.doctor_profile)
        
    db.delete(db_user)
    db.commit()
    return {"message": "User deleted successfully"}
