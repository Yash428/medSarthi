from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src import models, schemas
from src.database import get_db, engine
from src.dependencies import get_current_patient, get_current_doctor
from src.config import settings

from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent

router = APIRouter(prefix="/api/ai", tags=["ai"])

class ChatRequest(BaseModel):
    message: str

def get_llm():
    if settings.GROQ_API_KEY:
        return ChatGroq(temperature=0, groq_api_key=settings.GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    else:
        # Fallback to local Ollama
        return Ollama(base_url=settings.OLLAMA_BASE_URL, model="llama3")

@router.post("/patient/chat")
def patient_ai_chat(request: ChatRequest, current_user: models.User = Depends(get_current_patient)):
    llm = get_llm()
    # print(llm)
    system_prompt = "You are a helpful and empathetic AI Doctor for MedSarthi. Provide health advice but remind the patient to consult their real doctor."
    prompt = f"{system_prompt}\n\nPatient: {request.message}\nAI Doctor:"
    
    try:
        response = llm.invoke(prompt)
        # Handle both ChatModel and LLM outputs
        # print(response)
        content = response.content if hasattr(response, 'content') else str(response)
        return {"response": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI provider error: {str(e)}")

@router.post("/doctor/sql-query")
def doctor_sql_agent(request: ChatRequest, current_user: models.User = Depends(get_current_doctor)):
    try:
        llm = get_llm()
        # We need a synchronous engine for SQLDatabase, which engine from database.py is.
        db = SQLDatabase(engine)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="zero-shot-react-description",
            handle_parsing_errors=True
        )
        
        # Add context to enforce security: The AI should only describe data and not delete anything
        security_prompt = "You are an AI assistant for a doctor. You can only read data. Never run DROP or DELETE."
        input_query = f"{security_prompt}\nQuestion: {request.message}"
        
        response = agent_executor.invoke({"input": input_query})
        return {"response": response.get("output", "Could not determine an answer.")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

