from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional
from src import models, schemas
from src.database import get_db, engine
from src.dependencies import get_current_patient, get_current_doctor
from src.config import settings
from src.clinic_agent import create_clinic_agent
import re
import json

from langchain_groq import ChatGroq
from langchain_community.llms import Ollama

router = APIRouter(prefix="/api/ai", tags=["ai"])

class ChatRequest(BaseModel):
    message: str

class ClinicInsightRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []   # [{"role": "user"/"assistant", "content": "..."}]

def get_llm():
    if settings.GROQ_API_KEY:
        return ChatGroq(temperature=0, groq_api_key=settings.GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    else:
        return Ollama(base_url=settings.OLLAMA_BASE_URL, model="llama3")

@router.post("/patient/chat")
def patient_ai_chat(request: ChatRequest, current_user: models.User = Depends(get_current_patient)):
    llm = get_llm()
    system_prompt = "You are a helpful and empathetic AI Doctor for MedSarthi. Provide health advice but remind the patient to consult their real doctor."
    prompt = f"{system_prompt}\n\nPatient: {request.message}\nAI Doctor:"
    
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

        agent_executor = create_clinic_agent(doctor_id=doctor_id, db=db)
        
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

@router.post("/specialist/chat")
def specialist_ai_chat(request: SpecialistChatRequest, current_user: models.User = Depends(get_current_patient)):
    try:
        llm = get_llm()
        retriever = get_specialty_retriever(request.specialty)
        
        system_prompt = (
            f"You are a specialized AI Doctor focused on {request.specialty}. "
            "Use the following retrieved context from uploaded medical literature to answer the user's question. "
            "If the context doesn't have the answer, use your baseline medical knowledge but explicitly state that you are doing so. "
            "\n\nContext:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Patient: {input}\nAI Doctor:")
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": request.message})
        return {"response": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
