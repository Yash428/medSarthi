from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from src import models, schemas
from src.database import get_db
from src.dependencies import get_current_doctor

router = APIRouter(prefix="/api/doctor", tags=["doctor"])

@router.get("/profile", response_model=schemas.DoctorProfileResponse)
def get_profile(current_user: models.User = Depends(get_current_doctor), db: Session = Depends(get_db)):
    return current_user.doctor_profile

@router.get("/patients", response_model=List[schemas.PatientProfileResponse])
def get_patients(current_user: models.User = Depends(get_current_doctor), db: Session = Depends(get_db)):
    # Get patients who have appointments with this doctor
    patients = db.query(models.PatientProfile).join(models.Appointment).filter(models.Appointment.doctor_id == current_user.doctor_profile.id).distinct().all()
    return patients

@router.post("/prescriptions", response_model=schemas.PrescriptionResponse)
def create_prescription(
    prescription: schemas.PrescriptionCreate, 
    current_user: models.User = Depends(get_current_doctor), 
    db: Session = Depends(get_db)
):
    # Verify patient exists
    patient = db.query(models.PatientProfile).filter(models.PatientProfile.id == prescription.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    db_prescription = models.Prescription(
        patient_id=prescription.patient_id,
        doctor_id=current_user.doctor_profile.id,
        medicine_details=prescription.medicine_details,
        instructions=prescription.instructions
    )
    db.add(db_prescription)
    db.commit()
    db.refresh(db_prescription)
    return db_prescription

@router.get("/appointments", response_model=List[schemas.AppointmentResponse])
def get_appointments(current_user: models.User = Depends(get_current_doctor), db: Session = Depends(get_db)):
    return db.query(models.Appointment).filter(models.Appointment.doctor_id == current_user.doctor_profile.id).all()

@router.put("/profile", response_model=schemas.DoctorProfileResponse)
def update_profile(
    profile_update: schemas.DoctorProfileUpdate,
    current_user: models.User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    profile = current_user.doctor_profile
    if profile_update.specialization is not None:
        profile.specialization = profile_update.specialization
    if profile_update.experience_years is not None:
        profile.experience_years = profile_update.experience_years
    if profile_update.clinic_address is not None:
        profile.clinic_address = profile_update.clinic_address
    
    db.commit()
    db.refresh(profile)
    return profile

@router.put("/appointments/{appointment_id}", response_model=schemas.AppointmentResponse)
def update_appointment_status(
    appointment_id: int,
    appointment_update: schemas.AppointmentUpdate,
    current_user: models.User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    appointment = db.query(models.Appointment).filter(
        models.Appointment.id == appointment_id,
        models.Appointment.doctor_id == current_user.doctor_profile.id
    ).first()
    
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")
        
    appointment.status = appointment_update.status
    db.commit()
    db.refresh(appointment)
    return appointment

@router.get("/patients/{patient_id}/history")
def get_patient_history(
    patient_id: int,
    current_user: models.User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    # Verify the doctor has seen this patient
    has_access = db.query(models.Appointment).filter(
        models.Appointment.doctor_id == current_user.doctor_profile.id,
        models.Appointment.patient_id == patient_id
    ).first() is not None
    
    if not has_access:
        raise HTTPException(status_code=403, detail="Not authorized to view this patient's history")
        
    prescriptions = db.query(models.Prescription).filter(models.Prescription.patient_id == patient_id).all()
    reports = db.query(models.MedicalReport).filter(models.MedicalReport.patient_id == patient_id).all()
    
    return {
        "prescriptions": prescriptions,
        "reports": reports
    }

@router.get("/patients/{patient_id}/vitals", response_model=List[schemas.VitalLogResponse])
def get_patient_vitals(
    patient_id: int,
    current_user: models.User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    # Verify the doctor has seen this patient
    has_access = db.query(models.Appointment).filter(
        models.Appointment.doctor_id == current_user.doctor_profile.id,
        models.Appointment.patient_id == patient_id
    ).first() is not None

    if not has_access:
        raise HTTPException(status_code=403, detail="Not authorized to view this patient's vitals")

    return db.query(models.VitalLog)\
             .filter(models.VitalLog.patient_id == patient_id)\
             .order_by(models.VitalLog.recorded_at.desc())\
             .all()

