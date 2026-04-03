"""
clinic_agent.py — Agentic RAG engine for MedSarthi Doctor Insights
Provides a LangChain agent with 7 specialized, doctor-scoped tools
that reason over all clinical data in the database.

Uses create_openai_tools_agent (native function-calling) instead of
ReAct text-format, which avoids infinite loops on smaller LLMs.
"""

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


# ─────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────

def get_llm():
    """Return a Chat-compatible LLM that supports function/tool calling."""
    if settings.GROQ_API_KEY:
        return ChatGroq(
            temperature=0,
            groq_api_key=settings.GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
        )
    # Ollama with function-calling capable model
    return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model="llama3")


# ─────────────────────────────────────────────────────────────
# SCHEMA DESCRIPTION — given to the agent as context
# ─────────────────────────────────────────────────────────────

SCHEMA_HELP = "Tables: users, patient_profiles, doctor_profiles, appointments, prescriptions, medical_reports, vital_logs."


# ─────────────────────────────────────────────────────────────
# PYDANTIC INPUT SCHEMAS for each tool
# ─────────────────────────────────────────────────────────────

class PatientListInput(BaseModel):
    query: Optional[str] = Field(None, description="Optional search term or reason (e.g. 'all patients')")



class PatientInput(BaseModel):
    patient: str = Field(description="Patient name (partial match OK) or numeric patient ID")

class AppointmentFilterInput(BaseModel):
    filter: str = Field(
        default="",
        description="Filter string: a patient name, or one of 'upcoming', 'completed', 'cancelled'. Leave blank for all."
    )

class SqlInput(BaseModel):
    sql: str = Field(description="A valid PostgreSQL SELECT statement to execute against the clinic database")


# ─────────────────────────────────────────────────────────────
# TOOL FUNCTIONS — each scoped to the current doctor
# ─────────────────────────────────────────────────────────────

def make_tools(doctor_id: int, db: Session) -> List[Tool]:

    def get_my_patients(query: Optional[str] = None) -> str:
        """Return all patients linked to this doctor."""
        patients = (
            db.query(models.PatientProfile)
            .join(models.Appointment, models.Appointment.patient_id == models.PatientProfile.id)
            .filter(models.Appointment.doctor_id == doctor_id)
            .distinct()
            .all()
        )
        if not patients:
            return json.dumps({"error": "You currently have no patients linked to your account.", "data": []})
        data = []
        for p in patients:
            data.append({
                "patient_id": p.id,
                "username": p.user.username if p.user else "Unknown",
                "age": p.age,
                "gender": p.gender,
                "blood_group": p.blood_group
            })
        return json.dumps({"data": data, "count": len(data)})

    def get_patient_summary(patient: str) -> str:
        """Get full profile and medical history for a patient."""
        patient_obj = _resolve_patient(patient, doctor_id, db)
        if not patient_obj:
            return json.dumps({"error": f"Could not find a patient matching '{patient}' in your roster."})
        u = patient_obj.user
        return json.dumps({
            "profile": {
                "id": patient_obj.id,
                "username": u.username if u else "Unknown",
                "email": u.email if u else "N/A",
                "age": patient_obj.age,
                "gender": patient_obj.gender,
                "blood_group": patient_obj.blood_group,
                "medical_history_notes": patient_obj.medical_history
            }
        })

    def get_patient_prescriptions(patient: str) -> str:
        """Get all prescriptions issued to a patient."""
        patient_obj = _resolve_patient(patient, doctor_id, db)
        if not patient_obj:
            return json.dumps({"error": f"Could not find patient matching '{patient}'."})
        now = datetime.now()
        prescriptions = (
            db.query(models.Prescription)
            .filter(
                models.Prescription.patient_id == patient_obj.id,
                models.Prescription.medicine_details != None,
                models.Prescription.medicine_details != "",
                models.Prescription.created_at <= now
            )
            .order_by(models.Prescription.created_at.desc())
            .all()
        )
        if not prescriptions:
            return json.dumps({"error": f"No past medication records found for patient {patient_obj.user.username if patient_obj.user else patient_obj.id}.", "data": []})
        data = []
        for rx in prescriptions:
            data.append({
                "issued_on": rx.created_at.isoformat(),
                "medicine_details": rx.medicine_details,
                "instructions": rx.instructions
            })
        return json.dumps({"patient_name": patient_obj.user.username if patient_obj.user else "Patient", "prescriptions": data})

    def get_patient_vitals(patient: str) -> str:
        """Get blood pressure and blood sugar history for a patient."""
        patient_obj = _resolve_patient(patient, doctor_id, db)
        if not patient_obj:
            return json.dumps({"error": f"Could not find patient matching '{patient}'."})
        now = datetime.now()
        vitals = (
            db.query(models.VitalLog)
            .filter(
                models.VitalLog.patient_id == patient_obj.id,
                models.VitalLog.value != None,
                models.VitalLog.value != "",
                models.VitalLog.recorded_at <= now
            )
            .order_by(models.VitalLog.recorded_at.desc())
            .limit(20)
            .all()
        )
        if not vitals:
            return json.dumps({"error": "No vital signs logged yet.", "data": []})
        
        data = []
        for v in vitals:
            data.append({
                "type": v.vital_type,
                "value": v.value,
                "recorded_at": v.recorded_at.isoformat(),
                "notes": v.notes
            })
        return json.dumps({"patient": patient_obj.user.username if patient_obj.user else "Patient", "vitals": data})

    def get_patient_reports(patient: str) -> str:
        """Get all medical reports and diagnostic results for a patient."""
        patient_obj = _resolve_patient(patient, doctor_id, db)
        if not patient_obj:
            return json.dumps({"error": f"Could not find patient matching '{patient}'."})
        now = datetime.now()
        reports = (
            db.query(models.MedicalReport)
            .filter(
                models.MedicalReport.patient_id == patient_obj.id,
                models.MedicalReport.uploaded_at <= now
            )
            .order_by(models.MedicalReport.uploaded_at.desc())
            .all()
        )
        if not reports:
            return json.dumps({"error": "No past medical reports found.", "data": []})
        data = []
        for rep in reports:
            parsed_notes = None
            if rep.notes:
                try: parsed_notes = json.loads(rep.notes)
                except: parsed_notes = rep.notes
            data.append({
                "file_name": rep.file_name,
                "uploaded_at": rep.uploaded_at.isoformat(),
                "notes": parsed_notes
            })
        return json.dumps({"patient": patient_obj.user.username if patient_obj.user else "Patient", "reports": data})

    def get_appointments(filter: str = "") -> str:
        """Get appointment history."""
        query = (
            db.query(models.Appointment)
            .filter(models.Appointment.doctor_id == doctor_id)
            .order_by(models.Appointment.appointment_date.desc())
        )
        lower = filter.lower().strip()
        if lower in ("upcoming", "scheduled"):
            query = query.filter(models.Appointment.status == "SCHEDULED")
        elif lower == "completed":
            query = query.filter(models.Appointment.status == "COMPLETED")
        elif lower == "cancelled":
            query = query.filter(models.Appointment.status == "CANCELLED")
        elif filter.strip():
            patient = _resolve_patient(filter, doctor_id, db)
            if patient:
                query = query.filter(models.Appointment.patient_id == patient.id)

        appointments = query.limit(15).all()
        if not appointments:
            return json.dumps({"error": "No matching appointments found.", "data": []})

        data = []
        for a in appointments:
            data.append({
                "id": a.id,
                "date": a.appointment_date.isoformat(),
                "patient": a.patient.user.username if a.patient and a.patient.user else "Unknown",
                "status": a.status,
                "notes": a.notes
            })
        return json.dumps({"appointments": data})

    def run_safe_query(sql: str) -> str:
        """Execute a custom read-only SELECT SQL query against the database."""
        cleaned = sql.strip().rstrip(";")
        if not re.match(r"^\s*SELECT\b", cleaned, re.IGNORECASE):
            return "ERROR: Only SELECT queries are permitted. I cannot modify the database."
        # Inject doctor scope where possible
        try:
            result = db.execute(text(cleaned + " LIMIT 50"))
            rows = result.fetchall()
            cols = result.keys()
            if not rows:
                return "Query returned no results."
            header = " | ".join(str(c) for c in cols)
            divider = "-" * len(header)
            body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
            return f"{header}\n{divider}\n{body}"
        except Exception as e:
            return f"Query error: {str(e)}"

    return [
        StructuredTool.from_function(
            func=get_my_patients,
            name="get_my_patients",
            description="Get a list of all your patients with their basic details.",
            args_schema=PatientListInput,
        ),
        StructuredTool.from_function(
            func=get_patient_summary,
            name="get_patient_summary",
            description="Get the full profile and medical history for one patient.",
            args_schema=PatientInput,
        ),
        StructuredTool.from_function(
            func=get_patient_prescriptions,
            name="get_patient_prescriptions",
            description="Get all prescriptions issued to a specific patient.",
            args_schema=PatientInput,
        ),
        StructuredTool.from_function(
            func=get_patient_vitals,
            name="get_patient_vitals",
            description="Get blood pressure and blood sugar history with trend analysis and clinical flags for a patient.",
            args_schema=PatientInput,
        ),
        StructuredTool.from_function(
            func=get_patient_reports,
            name="get_patient_reports",
            description="Get all uploaded diagnostic reports and lab metrics for a patient.",
            args_schema=PatientInput,
        ),
        StructuredTool.from_function(
            func=get_appointments,
            name="get_appointments",
            description="Get appointment history. Optionally filter by patient name, or pass 'upcoming', 'completed', 'cancelled'.",
            args_schema=AppointmentFilterInput,
        ),
        StructuredTool.from_function(
            func=run_safe_query,
            name="run_safe_query",
            description="Run a custom SELECT SQL query for anything not covered by other tools.",
            args_schema=SqlInput,
        ),
    ]


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _resolve_patient(input_str: str, doctor_id: int, db: Session) -> Optional[models.PatientProfile]:
    """Find a patient by name (partial match) or ID, scoped to this doctor."""
    # Get all patients who have appointments with this doctor
    base_query = (
        db.query(models.PatientProfile)
        .join(models.Appointment, models.Appointment.patient_id == models.PatientProfile.id)
        .filter(models.Appointment.doctor_id == doctor_id)
        .distinct()
    )
    # Try numeric ID first
    stripped = input_str.strip()
    if stripped.isdigit():
        return base_query.filter(models.PatientProfile.id == int(stripped)).first()
    # Try username search
    patients = base_query.join(models.User, models.User.id == models.PatientProfile.user_id).all()
    lower = stripped.lower()
    for p in patients:
        if p.user and lower in p.user.username.lower():
            return p
    return None


def _analyze_trend(values: List[int]) -> str:
    """Analyze a list of numeric readings (most recent first) and return a trend description."""
    if len(values) < 2:
        return "Insufficient data for trend analysis."
    recent = values[:3]
    if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
        return "📈 Worsening — readings are consistently increasing."
    if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
        return "📉 Improving — readings are consistently decreasing."
    return "〰️  Fluctuating — no consistent direction."


# ─────────────────────────────────────────────────────────────
# AGENT FACTORY
# ─────────────────────────────────────────────────────────────

def create_clinic_agent(doctor_id: int, db: Session, language: str = "en") -> AgentExecutor:
    """Build a tool-calling agent executor for the given doctor, with multi-lingual support."""
    llm = get_llm()
    tools = make_tools(doctor_id, db)
    
    lang_map = {
        "hi": "Hindi",
        "gu": "Gujarati",
        "en": "English"
    }
    target_lang = lang_map.get(language, "English")

    system_content = (
        "You are MedSarthi Clinic Intelligence, a professional AI medical analyst for doctors. "
        f"MANDATORY: You MUST provide your entire final answer and clinical analysis in {target_lang}. "
        f"If the target language is Gujarati, do not use English even if the tools return data in English.\n\n"
        "GOAL:\n"
        "Instead of just listing data, provide a professional, synthesized clinical summary. "
        "Highlight significant trends, anomalies, and at-risk metrics (e.g., Blood Pressure >= 140/90 or Blood Sugar >= 126).\n\n"
        "STRICT TRUTH RULE:\n"
        "- If a tool returns a 'No data found' or an error message, report that CLEARLY.\n"
        "- NEVER invent or hallucinate medicine names, dosages, or trends if the data is not in the tool observation.\n"
        "- If medicine_details is empty or null, say 'No active medications found'.\n\n"
        "FORMATTING:\n"
        "- All tools now return structured JSON. Parse the JSON and present it professionally.\n"
        "- Use Markdown tables for vitals and longitudinal data.\n"
        "- Use bold text for critical findings.\n"
        "- Organize your response with clear headers (e.g., Clinical Summary, Vital Trends, Medications).\n\n"
        f"DATABASE OVERVIEW: {SCHEMA_HELP}\n\n"
        "CLINICAL RULES:\n"
        "- Always summarize trends if multiple readings are present.\n"
        "- Once you call a tool and receive data, provide a structured analytical FINAL ANSWER.\n"
        "- DO NOT call the same tool with the same parameters more than once.\n\n"
        "INSTRUCTIONS:\n"
        "1. Identify the patient/topic.\n"
        "2. Call the most relevant tool.\n"
        "3. Process the observation and give a professional clinical summary in the specified language."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_content),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # create_tool_calling_agent is more robust for Groq than create_openai_tools_agent
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
        max_execution_time=60,
    )
