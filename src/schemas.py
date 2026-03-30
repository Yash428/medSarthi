from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
from src.models import UserRole

# User Schemas
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    role: UserRole

class UserRegisterByDoctor(BaseModel):
    email: EmailStr
    username: str
    # role is implicitly patient

class UserResponse(BaseModel):
    id: int
    email: EmailStr
    username: str
    role: UserRole
    is_active: bool

    class Config:
        from_attributes = True

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[int] = None

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

# Business Schemas
class AppointmentCreate(BaseModel):
    doctor_id: int
    appointment_date: datetime
    notes: Optional[str] = None

class AppointmentUpdate(BaseModel):
    status: str

class AppointmentResponse(BaseModel):
    id: int
    patient_id: int
    doctor_id: int
    appointment_date: datetime
    status: str
    notes: Optional[str]

    class Config:
        from_attributes = True

class PrescriptionCreate(BaseModel):
    patient_id: int
    medicine_details: Optional[str] = None
    instructions: Optional[str] = None
    lab_orders: Optional[str] = None

class PrescriptionResponse(BaseModel):
    id: int
    patient_id: int
    doctor_id: int
    medicine_details: Optional[str]
    instructions: Optional[str]
    lab_orders: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class MedicalReportResponse(BaseModel):
    id: int
    patient_id: int
    file_name: str
    file_path: str
    uploaded_at: datetime
    notes: Optional[str]
    prescription_id: Optional[int] = None
    linked_test: Optional[str] = None

    class Config:
        from_attributes = True

class UserBasicInfo(BaseModel):
    username: str
    email: EmailStr

    class Config:
        from_attributes = True

class PatientProfileUpdate(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    blood_group: Optional[str] = None
    medical_history: Optional[str] = None

class PatientProfileResponse(BaseModel):
    id: int
    user_id: int
    user: UserBasicInfo
    age: Optional[int]
    gender: Optional[str]
    blood_group: Optional[str]
    medical_history: Optional[str]

    class Config:
        from_attributes = True

class DoctorProfileResponse(BaseModel):
    id: int
    user_id: int
    user: UserBasicInfo
    specialization: str
    experience_years: Optional[int]
    clinic_address: Optional[str]

    class Config:
        from_attributes = True

class DoctorProfileUpdate(BaseModel):
    specialization: Optional[str] = None
    experience_years: Optional[int] = None
    clinic_address: Optional[str] = None

class VitalLogCreate(BaseModel):
    vital_type: str  # "BP" or "BLOOD_SUGAR"
    value: str
    notes: Optional[str] = None

class VitalLogResponse(BaseModel):
    id: int
    patient_id: int
    vital_type: str
    value: str
    notes: Optional[str]
    recorded_at: datetime

    class Config:
        from_attributes = True

# AI Lab Recommendation Schemas
class LabRecommendationRequest(BaseModel):
    patient_id: Optional[int] = None
    symptoms: str
    current_orders: Optional[List[str]] = []

class LabRecommendation(BaseModel):
    test_name: str
    reason: str

class LabRecommendationResponse(BaseModel):
    recommendations: List[LabRecommendation]
