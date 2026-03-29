import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
from src import models, schemas
from src.database import get_db
from src.dependencies import get_current_patient

router = APIRouter(prefix="/api/patient", tags=["patient"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/profile", response_model=schemas.PatientProfileResponse)
def get_profile(current_user: models.User = Depends(get_current_patient), db: Session = Depends(get_db)):
    return current_user.patient_profile

@router.put("/profile", response_model=schemas.PatientProfileResponse)
def update_profile(
    profile_update: schemas.PatientProfileUpdate,
    current_user: models.User = Depends(get_current_patient),
    db: Session = Depends(get_db)
):
    profile = current_user.patient_profile
    if profile_update.age is not None:
        profile.age = profile_update.age
    if profile_update.gender is not None:
        profile.gender = profile_update.gender
    if profile_update.blood_group is not None:
        profile.blood_group = profile_update.blood_group
    if profile_update.medical_history is not None:
        profile.medical_history = profile_update.medical_history
    
    db.commit()
    db.refresh(profile)
    return profile

@router.post("/appointments", response_model=schemas.AppointmentResponse)
def book_appointment(appointment: schemas.AppointmentCreate, current_user: models.User = Depends(get_current_patient), db: Session = Depends(get_db)):
    db_appointment = models.Appointment(
        patient_id=current_user.patient_profile.id,
        doctor_id=appointment.doctor_id,
        appointment_date=appointment.appointment_date,
        notes=appointment.notes
    )
    db.add(db_appointment)
    db.commit()
    db.refresh(db_appointment)
    return db_appointment

@router.get("/appointments", response_model=List[schemas.AppointmentResponse])
def get_appointments(current_user: models.User = Depends(get_current_patient), db: Session = Depends(get_db)):
    return db.query(models.Appointment).filter(models.Appointment.patient_id == current_user.patient_profile.id).all()

@router.get("/prescriptions", response_model=List[schemas.PrescriptionResponse])
def get_prescriptions(current_user: models.User = Depends(get_current_patient), db: Session = Depends(get_db)):
    return db.query(models.Prescription).filter(models.Prescription.patient_id == current_user.patient_profile.id).all()

@router.post("/reports", response_model=schemas.MedicalReportResponse)
def upload_report(
    file: UploadFile = File(...), 
    notes: str = Form(None),
    current_user: models.User = Depends(get_current_patient), 
    db: Session = Depends(get_db)
):
    file_path = os.path.join(UPLOAD_DIR, f"{current_user.id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    report = models.MedicalReport(
        patient_id=current_user.patient_profile.id,
        file_name=file.filename,
        file_path=file_path,
        notes=notes
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report

@router.put("/appointments/{appointment_id}", response_model=schemas.AppointmentResponse)
def update_appointment(
    appointment_id: int, 
    appointment_update: schemas.AppointmentUpdate,
    current_user: models.User = Depends(get_current_patient), 
    db: Session = Depends(get_db)
):
    appointment = db.query(models.Appointment).filter(
        models.Appointment.id == appointment_id,
        models.Appointment.patient_id == current_user.patient_profile.id
    ).first()
    
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")
        
    appointment.status = appointment_update.status
    db.commit()
    db.refresh(appointment)
    return appointment

@router.get("/reports", response_model=List[schemas.MedicalReportResponse])
def get_reports(current_user: models.User = Depends(get_current_patient), db: Session = Depends(get_db)):
    return db.query(models.MedicalReport).filter(models.MedicalReport.patient_id == current_user.patient_profile.id).all()

@router.get("/history")
def get_history(current_user: models.User = Depends(get_current_patient), db: Session = Depends(get_db)):
    prescriptions = db.query(models.Prescription).filter(models.Prescription.patient_id == current_user.patient_profile.id).all()
    reports = db.query(models.MedicalReport).filter(models.MedicalReport.patient_id == current_user.patient_profile.id).all()
    
    return {
        "prescriptions": prescriptions,
        "reports": reports
    }

@router.get("/doctors", response_model=List[schemas.DoctorProfileResponse])
def get_all_doctors(current_user: models.User = Depends(get_current_patient), db: Session = Depends(get_db)):
    return db.query(models.DoctorProfile).all()

@router.post("/vitals", response_model=schemas.VitalLogResponse)
def log_vital(
    vital: schemas.VitalLogCreate,
    current_user: models.User = Depends(get_current_patient),
    db: Session = Depends(get_db)
):
    log = models.VitalLog(
        patient_id=current_user.patient_profile.id,
        vital_type=vital.vital_type,
        value=vital.value,
        notes=vital.notes
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log

@router.get("/vitals", response_model=List[schemas.VitalLogResponse])
def get_vitals(current_user: models.User = Depends(get_current_patient), db: Session = Depends(get_db)):
    return db.query(models.VitalLog)\
             .filter(models.VitalLog.patient_id == current_user.patient_profile.id)\
             .order_by(models.VitalLog.recorded_at.desc())\
             .all()
