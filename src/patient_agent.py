import json
import re
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import text
from langchain.tools import Tool, StructuredTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field

from src import models
from src.config import settings

def get_llm():
    if settings.GROQ_API_KEY:
        return ChatGroq(
            temperature=0,
            groq_api_key=settings.GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
        )
    return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model="llama3")

def make_patient_tools(patient_id: int, db: Session, language: str = "en") -> List[Tool]:
    """Provides tools for a patient to view their own clinical data."""
    
    lang_map = {"hi": "Hindi", "gu": "Gujarati", "en": "English"}
    target_lang = lang_map.get(language, "English")
    
    # Text fragments for different languages
    labels = {
        "hi": {
            "vitals": "आपके हालिया वाइटल संकेत:",
            "rxs": "आपके पास {n} प्रिस्क्रिप्शन हैं:",
            "reports": "आपकी मेडिकल रिपोर्ट ({n} कुल):",
            "appointments": "आपकी हालिया नियुक्तियां:",
            "disclaimer": "\n\nकृपया उचित निदान के लिए वास्तविक डॉक्टर से परामर्श लें।"
        },
        "gu": {
            "vitals": "તમારા તાજેતરના મહત્વના સંકેતો:",
            "rxs": "તમારી પાસે {n} પ્રિસ્ક્રિપ્શન છે:",
            "reports": "તમારા મેડિકલ રિપોર્ટ્સ ({n} કુલ):",
            "appointments": "તમારી તાજેતરની મુલાકાતો:",
            "disclaimer": "\n\nકૃપા કરીને યોગ્ય નિદાન માટે વાસ્તવિક ડૉક્ટરની સલાહ લો."
        },
        "en": {
            "vitals": "Your Recent Vital Signs:",
            "rxs": "You have {n} prescription(s):",
            "reports": "Your Medical Reports ({n} total):",
            "appointments": "Your Recent Appointments:",
            "disclaimer": "\n\n*Please consult a real doctor for clinical diagnosis.*"
        }
    }.get(language, labels["en"] if "labels" in locals() else {}) # Fallback to English
    
    # Ensure fallback labels exist
    if not labels or "vitals" not in labels:
        labels = {
            "vitals": "Your Recent Vital Signs:",
            "rxs": "You have {n} prescription(s):",
            "reports": "Your Medical Reports ({n} total):",
            "appointments": "Your Recent Appointments:",
            "disclaimer": "\n\n*Please consult a real doctor for clinical diagnosis.*"
        }

    def get_my_vitals(query: Optional[str] = None) -> str:
        """Get your blood pressure and blood sugar history with trend analysis."""
        now = datetime.now()
        vitals = (
            db.query(models.VitalLog)
            .filter(
                models.VitalLog.patient_id == patient_id,
                models.VitalLog.value != None,
                models.VitalLog.value != "",
                models.VitalLog.recorded_at <= now
            )
            .order_by(models.VitalLog.recorded_at.desc())
            .limit(10)
            .all()
        )
        if not vitals:
            return json.dumps({"error": "No past vital signs found.", "data": []})
        
        data = []
        for v in vitals:
            data.append({
                "recorded_at": v.recorded_at.isoformat(),
                "vital_type": v.vital_type,
                "value": v.value,
                "notes": v.notes
            })
        
        return json.dumps({"data": data, "patient_id": patient_id})

    def get_my_prescriptions(query: Optional[str] = None) -> str:
        """Get a list of all your active and past prescriptions."""
        now = datetime.now()
        prescriptions = (
            db.query(models.Prescription)
            .filter(
                models.Prescription.patient_id == patient_id,
                models.Prescription.medicine_details != None,
                models.Prescription.medicine_details != "",
                models.Prescription.created_at <= now
            )
            .order_by(models.Prescription.created_at.desc())
            .all()
        )
        if not prescriptions:
            return json.dumps({"error": "No past prescriptions found with medication records.", "data": []})
        
        data = []
        for rx in prescriptions:
            data.append({
                "issued_on": rx.created_at.isoformat(),
                "medicine_details": rx.medicine_details,
                "instructions": rx.instructions
            })
        return json.dumps({"data": data, "patient_id": patient_id})

    def get_my_medical_reports(query: Optional[str] = None) -> str:
        """Get a list of your uploaded medical and diagnostic reports."""
        now = datetime.now()
        reports = (
            db.query(models.MedicalReport)
            .filter(
                models.MedicalReport.patient_id == patient_id,
                models.MedicalReport.uploaded_at <= now
            )
            .order_by(models.MedicalReport.uploaded_at.desc())
            .all()
        )
        if not reports:
            return json.dumps({"error": "No past medical reports found.", "data": []})
        
        data = []
        for rep in reports:
            notes_parsed = {}
            if rep.notes:
                try:
                    notes_parsed = json.loads(rep.notes)
                except:
                    notes_parsed = {"raw_notes": rep.notes}
            data.append({
                "file_name": rep.file_name,
                "uploaded_at": rep.uploaded_at.isoformat(),
                "insights": notes_parsed
            })
        return json.dumps({"data": data, "patient_id": patient_id})

    def get_my_appointments(query: Optional[str] = None) -> str:
        """Get your upcoming and past appointments details."""
        appointments = (
            db.query(models.Appointment)
            .filter(models.Appointment.patient_id == patient_id)
            .order_by(models.Appointment.appointment_date.desc())
            .limit(10)
            .all()
        )
        if not appointments:
            return json.dumps({"error": "No appointments found", "data": []})
        
        data = []
        for a in appointments:
            doc_name = a.doctor.user.username if a.doctor and a.doctor.user else "Unknown Doctor"
            data.append({
                "appointment_date": a.appointment_date.isoformat(),
                "doctor_name": doc_name,
                "status": a.status
            })
        return json.dumps({"data": data, "patient_id": patient_id})

    class QueryInput(BaseModel):
        query: Optional[str] = Field(None, description="The user's query description")

    return [
        StructuredTool.from_function(
            func=get_my_vitals,
            name="get_my_vitals",
            description="Fetch your logged vital signs like BP, Sugar, Heart Rate.",
            args_schema=QueryInput,
        ),
        StructuredTool.from_function(
            func=get_my_prescriptions,
            name="get_my_prescriptions",
            description="Fetch all medicines and instructions prescribed to you.",
            args_schema=QueryInput,
        ),
        StructuredTool.from_function(
            func=get_my_medical_reports,
            name="get_my_medical_reports",
            description="Fetch your uploaded lab reports, radiology findings, and files.",
            args_schema=QueryInput,
        ),
        StructuredTool.from_function(
            func=get_my_appointments,
            name="get_my_appointments",
            description="Fetch your appointment history and upcoming visit schedule.",
            args_schema=QueryInput,
        ),
    ]

def create_patient_agent(patient_id: int, db: Session, language: str = "en") -> AgentExecutor:
    """Build a tool-calling agent executor for the patient, with multi-lingual support."""
    llm = get_llm()
    tools = make_patient_tools(patient_id, db, language)
    
    lang_map = {"hi": "Hindi", "gu": "Gujarati", "en": "English"}
    target_lang = lang_map.get(language, "English")

    system_content = (
        "You are MedSarthi Health Insights, your personal health AI assistant. "
        "Your role is to help the user understand their medical records accurately and empathetically.\n\n"
        "RESPONSE HUMANIZER GUIDELINES:\n"
        "When you receive structured JSON data from a tool, convert it into a friendly, conversational response in " + target_lang + ".\n\n"
        "GOAL:\n"
        "- Briefly explain the condition or purpose of the record.\n"
        "- 💊 MEDICINES: List medicines with bullet points. Explain purposes and convert timings like '1-0-1' into natural phrases (e.g., 'Morning and Night').\n"
        "- 🩺 VITALS: Provide a friendly one-sentence summary of vital signs.\n"
        "- ⚠️ ADVICE: Include a clear 'Important Advice' section for dietary/lifestyle notes.\n"
        "- EMPATHY: Use supportive language. Avoid robotic listing.\n\n"
        "FORMATTING:\n"
        "- Use clear headers and bullet points.\n"
        f"- MANDATORY: You MUST provide all responses in {target_lang}.\n"
        f"  If the target language is Gujarati, do not use English even if the tool results provide it.\n\n"
        f"MANDATORY DISCLAIMER (Translate to {target_lang}):\n"
        "'If your symptoms worsen, please consult your doctor immediately.'\n\n"
        f"DATABASE ACCESS: Current Patient ID is {patient_id}.\n"
        "INSTRUCTIONS:\n"
        "1. Identify the requested health record.\n"
        "2. Retrieve the data using the matching tool (which returns JSON).\n"
        "3. Apply the RESPONSE HUMANIZER GUIDELINES to generate the final answer."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_content),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )
