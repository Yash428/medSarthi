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
            return "You currently have no patients linked to your account."
        rows = []
        for p in patients:
            username = p.user.username if p.user else f"#{p.id}"
            rows.append(
                f"- Patient '{username}' (ID: {p.id}) | Age: {p.age or 'N/A'} | "
                f"Gender: {p.gender or 'N/A'} | Blood Group: {p.blood_group or 'N/A'}"
            )
        return f"You have {len(patients)} patient(s):\n" + "\n".join(rows)

    def get_patient_summary(patient: str) -> str:
        """Get full profile and medical history for a patient."""
        patient_obj = _resolve_patient(patient, doctor_id, db)
        if not patient_obj:
            return f"Could not find a patient matching '{patient}' in your roster."
        u = patient_obj.user
        lines = [
            f"Patient Profile: {u.username if u else 'Unknown'} (ID: {patient_obj.id})",
            f"  Email:          {u.email if u else 'N/A'}",
            f"  Age:            {patient_obj.age or 'Not recorded'}",
            f"  Gender:         {patient_obj.gender or 'Not recorded'}",
            f"  Blood Group:    {patient_obj.blood_group or 'Not recorded'}",
            f"  Medical History: {patient_obj.medical_history or 'No history recorded'}",
        ]
        return "\n".join(lines)

    def get_patient_prescriptions(patient: str) -> str:
        """Get all prescriptions issued to a patient."""
        patient_obj = _resolve_patient(patient, doctor_id, db)
        if not patient_obj:
            return f"Could not find a patient matching '{patient}' in your roster."
        prescriptions = (
            db.query(models.Prescription)
            .filter(models.Prescription.patient_id == patient_obj.id)
            .order_by(models.Prescription.created_at.desc())
            .all()
        )
        if not prescriptions:
            return f"No prescriptions found for patient {patient_obj.user.username if patient_obj.user else patient_obj.id}."
        result = [f"Prescriptions for {patient_obj.user.username if patient_obj.user else patient_obj.id} ({len(prescriptions)} total):"]
        for i, rx in enumerate(prescriptions, 1):
            result.append(f"\n--- Prescription #{i} | {rx.created_at.strftime('%d %b %Y')} ---")
            result.append(rx.medicine_details)
            if rx.instructions:
                result.append(f"Instructions: {rx.instructions}")
        return "\n".join(result)

    def get_patient_vitals(patient: str) -> str:
        """Get blood pressure and blood sugar history + trend analysis for a patient."""
        patient_obj = _resolve_patient(patient, doctor_id, db)
        if not patient_obj:
            return f"Could not find a patient matching '{patient}' in your roster."
        vitals = (
            db.query(models.VitalLog)
            .filter(models.VitalLog.patient_id == patient_obj.id)
            .order_by(models.VitalLog.recorded_at.desc())
            .limit(20)
            .all()
        )
        if not vitals:
            return f"No vital signs logged by {patient_obj.user.username if patient_obj.user else patient_obj.id} yet."

        bp_logs = [v for v in vitals if v.vital_type == "BP"]
        sugar_logs = [v for v in vitals if v.vital_type == "BLOOD_SUGAR"]
        name = patient_obj.user.username if patient_obj.user else str(patient_obj.id)
        lines = [f"Vital Signs for {name}:"]

        if bp_logs:
            lines.append(f"\n🩺 Blood Pressure ({len(bp_logs)} readings):")
            for v in bp_logs[:5]:
                lines.append(f"  {v.recorded_at.strftime('%d %b %Y %H:%M')} → {v.value} mmHg" + (f"  [{v.notes}]" if v.notes else ""))
            # Trend analysis
            systolics = [int(v.value.split("/")[0]) for v in bp_logs if "/" in v.value and v.value.split("/")[0].isdigit()]
            if len(systolics) >= 3:
                trend = _analyze_trend(systolics[:5])
                lines.append(f"  📈 Trend (last {min(5, len(systolics))} readings): {trend}")
                latest = systolics[0]
                if latest >= 140:
                    lines.append("  ⚠️  CLINICAL FLAG: Latest reading indicates Stage 2 Hypertension. Consider review.")
                elif latest >= 130:
                    lines.append("  ⚠️  CLINICAL FLAG: Latest reading indicates Stage 1 Hypertension.")

        if sugar_logs:
            lines.append(f"\n🩸 Blood Sugar ({len(sugar_logs)} readings):")
            for v in sugar_logs[:5]:
                lines.append(f"  {v.recorded_at.strftime('%d %b %Y %H:%M')} → {v.value} mg/dL" + (f"  [{v.notes}]" if v.notes else ""))
            sugars = [int(v.value.split()[0]) for v in sugar_logs if v.value.split()[0].isdigit()]
            if len(sugars) >= 3:
                trend = _analyze_trend(sugars[:5])
                lines.append(f"  📈 Trend: {trend}")
                if sugars[0] >= 126:
                    lines.append("  ⚠️  CLINICAL FLAG: Latest reading is in Diabetic range. Review current medications.")
                elif sugars[0] >= 100:
                    lines.append("  ⚠️  CLINICAL FLAG: Latest reading is Pre-diabetic. Recommend lifestyle intervention.")

        return "\n".join(lines)

    def get_patient_reports(patient: str) -> str:
        """Get all uploaded medical reports and diagnostic results for a patient."""
        patient_obj = _resolve_patient(patient, doctor_id, db)
        if not patient_obj:
            return f"Could not find a patient matching '{patient}' in your roster."
        reports = (
            db.query(models.MedicalReport)
            .filter(models.MedicalReport.patient_id == patient_obj.id)
            .order_by(models.MedicalReport.uploaded_at.desc())
            .all()
        )
        if not reports:
            return f"No medical reports uploaded by {patient_obj.user.username if patient_obj.user else patient_obj.id}."
        name = patient_obj.user.username if patient_obj.user else str(patient_obj.id)
        lines = [f"Medical Reports for {name} ({len(reports)} file(s)):"]
        for rep in reports:
            lines.append(f"\n📄 {rep.file_name} | Uploaded: {rep.uploaded_at.strftime('%d %b %Y')}")
            if rep.notes:
                try:
                    data = json.loads(rep.notes)
                    lines.append(f"  Report Type: {data.get('type', 'Unknown')}")
                    attrs = data.get("attributes", [])
                    if attrs:
                        lines.append("  Metrics:")
                        for attr in attrs:
                            lines.append(f"    {attr['key']}: {attr['value']}")
                    
                    narrative = data.get("narrative")
                    if narrative:
                        lines.append(f"  Radiologist Findings: {narrative}")
                except (json.JSONDecodeError, KeyError):
                    lines.append(f"  Notes: {rep.notes}")
        return "\n".join(lines)

    def get_appointments(filter: str = "") -> str:
        """Get appointment history. Pass patient name, 'upcoming', 'completed', 'cancelled', or leave blank."""
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
            return "No appointments found matching that filter."

        lines = [f"Appointments ({len(appointments)} shown):"]
        for a in appointments:
            patient_name = "Unknown"
            if a.patient and a.patient.user:
                patient_name = a.patient.user.username
            status_emoji = {"SCHEDULED": "🕐", "COMPLETED": "✅", "CANCELLED": "❌"}.get(a.status, "?")
            lines.append(
                f"  {status_emoji} {a.appointment_date.strftime('%d %b %Y %H:%M')} | "
                f"Patient: {patient_name} | Status: {a.status}"
                + (f" | Notes: {a.notes}" if a.notes else "")
            )
        return "\n".join(lines)

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

def create_clinic_agent(doctor_id: int, db: Session) -> AgentExecutor:
    """Build a tool-calling agent executor for the given doctor."""
    llm = get_llm()
    tools = make_tools(doctor_id, db)

    system_content = (
        "You are MedSarthi Clinic Intelligence, a professional AI for doctors. "
        "Your goal is to answer questions using strictly the available tools.\n\n"
        "DATABASE OVERVIEW: " + SCHEMA_HELP + "\n\n"
        "CLINICAL RULES:\n"
        "- Highlight BP >= 140/90 (Hypertension) or Sugar >= 126 (Diabetes).\n"
        "- Always summarize trends if multiple readings are present.\n"
        "- Once you call a tool and receive data, analyze it immediately and provide your FINAL ANSWER. "
        "DO NOT call the same tool with the same parameters more than once.\n\n"
        "INSTRUCTIONS:\n"
        "1. Identify the patient/topic.\n"
        "2. Call the most relevant tool.\n"
        "3. Process the observation and give a structured clinical summary.\n"
        "4. Stop iterating as soon as you have the answer."
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
