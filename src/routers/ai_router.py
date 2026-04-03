from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
import tempfile
from src import models, schemas
from src.database import get_db
from src.dependencies import get_current_patient, get_current_doctor, get_current_user
from src.config import settings
from src.clinic_agent import create_clinic_agent
from src.patient_agent import create_patient_agent
from src.stt_service import stt_engine
import re
import json

from langchain_groq import ChatGroq
from langchain_community.llms import Ollama

router = APIRouter(prefix="/api/ai", tags=["ai"])

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "en"

class ClinicInsightRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []   # [{"role": "user"/"assistant", "content": "..."}]
    language: Optional[str] = "en"

def get_llm():
    if settings.GROQ_API_KEY:
        return ChatGroq(temperature=0, groq_api_key=settings.GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    else:
        return Ollama(base_url=settings.OLLAMA_BASE_URL, model="llama3")

@router.post("/patient/chat")
def patient_ai_chat(request: ChatRequest, current_user: models.User = Depends(get_current_patient)):
    llm = get_llm()
    
    lang_map = {
        "hi": "Hindi",
        "gu": "Gujarati",
        "en": "English"
    }
    target_lang = lang_map.get(request.language, "English")
    
    system_prompt = (
        f"You are a helpful and empathetic AI Doctor for MedSarthi. "
        f"MANDATORY: You MUST provide your entire response in {target_lang}. "
        f"If the target language is Gujarati, do not use English even if the user asks in English.\n\n"
        "GOAL:\n"
        "Provide informative medical advice and helpful explanations. "
        "Use Markdown formatting (bullet points, bolding) to make your response extremely readable. "
        "Remind the patient clearly to consult their real doctor for a clinical diagnosis."
    )
    prompt = f"{system_prompt}\n\nPatient: {request.message}\nAI Doctor ({target_lang}):"
    
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        return {"response": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI provider error: {str(e)}")

@router.post("/doctor/chat")
def doctor_ai_chat(request: ChatRequest, current_user: models.User = Depends(get_current_doctor)):
    """General AI assistant for doctors."""
    llm = get_llm()
    
    lang_map = {"hi": "Hindi", "gu": "Gujarati", "en": "English"}
    target_lang = lang_map.get(request.language, "English")
    
    system_prompt = (
        f"You are a professional medical research assistant for MedSarthi doctors. "
        f"MANDATORY: You MUST provide your entire response in {target_lang}. "
        f"If the user is in Gujarati mode, every word of your response must be in Gujarati.\n\n"
        "GOAL:\n"
        "Provide detailed medical knowledge, dosages, and research using professional terminology. "
        "Use Markdown for structured data and analysis. High accuracy and professional readability are key."
    )
    prompt = f"{system_prompt}\n\nDoctor: {request.message}\nAssistant ({target_lang}):"
    
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        return {"response": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI provider error: {str(e)}")


@router.post("/doctor/insights")
def doctor_clinic_insights(
    request: ClinicInsightRequest,
    current_user: models.User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """
    Agentic RAG endpoint for doctor clinical insights.
    Uses specialized tools scoped to the current doctor's patients.
    Accepts conversation history for contextual follow-up questions.
    """
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        doctor_id = current_user.doctor_profile.id

        # Convert history list → LangChain message objects for MessagesPlaceholder
        chat_history = []
        for msg in (request.history or []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                chat_history.append(HumanMessage(content=content))
            elif role == "assistant":
                chat_history.append(AIMessage(content=content))

        agent_executor = create_clinic_agent(doctor_id=doctor_id, db=db, language=request.language)
        
        result = agent_executor.invoke({
            "input": request.message,
            "chat_history": chat_history,
        })

        if not result:
            return {"response": "The intelligence agent did not respond. Please try again with a specific question."}

        # If the agent hit max iterations, it might have content in intermediate_steps
        # but LangChain usually puts a summary in 'output' even then.
        output = result.get("output")
        if not output:
             # Fallback: if no output but we have intermediate steps, the agent was close
             return {"response": "The agent was unable to finalize an answer within the iteration limit. Please try asking about a specific patient."}

        return {"response": output}

    except Exception as e:
        print(f"[clinic_agent] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Insight agent error: {str(e)}")

@router.post("/patient/insights")
def patient_health_insights(
    request: ClinicInsightRequest,
    current_user: models.User = Depends(get_current_patient),
    db: Session = Depends(get_db)
):
    """
    Agentic RAG endpoint for patients to query their own medical data.
    """
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        patient_id = current_user.patient_profile.id

        chat_history = []
        for msg in (request.history or []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                chat_history.append(HumanMessage(content=content))
            elif role == "assistant":
                chat_history.append(AIMessage(content=content))

        agent_executor = create_patient_agent(patient_id=patient_id, db=db, language=request.language)
        
        result = agent_executor.invoke({
            "input": request.message,
            "chat_history": chat_history,
        })

        return {"response": result.get("output", "I was unable to retrieve your health data. Please try again.")}

    except Exception as e:
        print(f"[patient_agent] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Insight agent error: {str(e)}")


# Keep the old sql-query endpoint as a legacy alias for compatibility
@router.post("/doctor/sql-query")
def doctor_sql_agent_legacy(
    request: ChatRequest,
    current_user: models.User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """Legacy endpoint — proxies to the new agentic system."""
    from pydantic import BaseModel
    forwarded = ClinicInsightRequest(message=request.message, history=[])
    return doctor_clinic_insights(forwarded, current_user, db)


# ─────────────────────────────────────────────────────────────
# Specialist RAG (unchanged)
# ─────────────────────────────────────────────────────────────

@router.post("/doctor/suggest-lab-orders", response_model=schemas.LabRecommendationResponse)
def suggest_lab_orders(
    request: schemas.LabRecommendationRequest,
    current_user: models.User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """
    Suggest lab tests and diagnostic investigations based on symptoms and current context.
    """
    llm = get_llm()
    
    patient_context = ""
    if request.patient_id:
        patient = db.query(models.PatientProfile).filter(models.PatientProfile.id == request.patient_id).first()
        if patient:
            patient_context = f"Patient Age: {patient.age or 'N/A'}, Gender: {patient.gender or 'N/A'}, History: {patient.medical_history or 'None'}. "

    current_orders_str = ", ".join(request.current_orders) if request.current_orders else "None"

    system_prompt = (
        "You are a Clinical Diagnostic Assistant. Your task is to recommend laboratory tests, "
        "imaging, or other diagnostic investigations based on the provided symptoms and patient context.\n"
        "Format: Return ONLY a JSON object with a 'recommendations' key containing a list of objects with 'test_name' and 'reason'.\n"
        "Example: {\"recommendations\": [{\"test_name\": \"CBC\", \"reason\": \"To check for anemia or infection due to fatigue\"}]}"
    )
    
    user_prompt = (
        f"Context: {patient_context}\n"
        f"Symptoms: {request.symptoms}\n"
        f"Current Orders: {current_orders_str}\n"
        "Suggest relevant tests with reasons."
    )

    try:
        response = llm.invoke(f"{system_prompt}\n\nUser: {user_prompt}")
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON if LLM yaps
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)
            
        import json
        data = json.loads(content)
        return data
    except Exception as e:
        print(f"[suggest_lab_orders] Error: {e}")
        return {"recommendations": []}

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.rag_manager import get_specialty_retriever

class SpecialistChatRequest(BaseModel):
    message: str
    specialty: str
    language: Optional[str] = "en"

@router.post("/specialist/chat")
def specialist_ai_chat(request: SpecialistChatRequest, current_user: models.User = Depends(get_current_patient)):
    try:
        llm = get_llm()
        retriever = get_specialty_retriever(request.specialty)
        
        lang_map = {
            "hi": "Hindi",
            "gu": "Gujarati",
            "en": "English"
        }
        target_lang = lang_map.get(request.language, "English")

        system_prompt = (
            f"You are a specialized AI Doctor focused on {request.specialty}. "
            f"MANDATORY: You MUST respond in {target_lang}. "
            f"Even if the query is in English, if the target language is Gujarati, you MUST respond in Gujarati. "
            "Use the following retrieved context from uploaded medical literature to answer the user's question. "
            "If the context doesn't have the answer, use your baseline medical knowledge but explicitly state that you are doing so. "
            "\n\nContext:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_content if 'system_content' in locals() else system_prompt),
            ("human", f"Patient ({target_lang}): {{input}}\nAI Doctor ({target_lang}):")
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": request.message})
        return {"response": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voice/stt")
async def speech_to_text(
    audio: UploadFile = File(...),
    language: Optional[str] = None,
    current_user: models.User = Depends(get_current_user)
):
    """
    Transcribe uploaded audio file to text.
    'language' can be 'hi', 'gu', 'en' etc.
    """
    try:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            shutil.copyfileobj(audio.file, temp_audio)
            temp_path = temp_audio.name

        # Transcribe
        text = stt_engine.transcribe(temp_path, language=language)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {"text": text}
    except Exception as e:
        print(f"[STT] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
