from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from src import models, schemas, auth
from src.database import get_db
from datetime import timedelta, datetime, timezone
import uuid
from src.utils.email_service import send_password_reset_email
from src.config import settings
from src.models import UserRole

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/signup", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    db_user = db.query(models.User).filter((models.User.email == user.email) | (models.User.username == user.username)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email or Username already registered")
    
    hashed_password = auth.get_password_hash(user.password)
    new_user = models.User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        role=user.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Automatically create respective profile
    if user.role == UserRole.PATIENT:
        profile = models.PatientProfile(user_id=new_user.id)
        db.add(profile)
    elif user.role == UserRole.DOCTOR:
        profile = models.DoctorProfile(user_id=new_user.id, specialization="General Physician")
        db.add(profile)
    
    db.commit()
    return new_user

@router.post("/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Credentials")
    
    if not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Credentials")
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"user_id": user.id, "role": user.role.value}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/forgot-password")
def forgot_password(req: schemas.ForgotPasswordRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == req.email).first()
    if not user:
        # For security, we don't leak account existence, just return success
        return {"message": "If the account exists, a reset link will be sent."}
    
    # Generate token
    token = str(uuid.uuid4())
    user.reset_token = token
    user.reset_token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    
    db.commit()
    
    # "Send" email asynchronously
    background_tasks.add_task(send_password_reset_email, user.email, token)
    
    return {"message": "Success! A reset link has been emailed."}

@router.post("/reset-password")
def reset_password(req: schemas.ResetPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(
        models.User.reset_token == req.token,
        models.User.reset_token_expiry > datetime.now(timezone.utc)
    ).first()
    
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    
    user.hashed_password = auth.get_password_hash(req.new_password)
    user.reset_token = None
    user.reset_token_expiry = None
    
    db.commit()
    return {"message": "Password reset successfully. You can now log in."}
