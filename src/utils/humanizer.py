import json
from typing import Optional
from src.patient_agent import get_llm

def humanize_medical_response(data: dict, language_code: str = "en") -> str:
    """
    Converts structured medical data into a friendly, conversational response.
    Specifically designed for MedSarthi users in English, Hindi, and Gujarati.
    """
    llm = get_llm()
    
    lang_map = {
        "hi": "Hindi",
        "gu": "Gujarati",
        "en": "English"
    }
    target_lang = lang_map.get(language_code, "English")
    
    # System prompt based on user requirements
    system_prompt = (
        f"You are a helpful medical assistant for MedSarthi. "
        f"Your task is to convert structured medical data (JSON) into a friendly, conversational response in {target_lang}.\n\n"
        "STYLE GUIDELINES:\n"
        "- Use simple and clear language.\n"
        "- Explain medicines clearly (e.g., purpose, timing, food).\n"
        "- Convert timings like '1-0-1' into natural language (e.g., 'Morning and Night').\n"
        "- Use bullet points for readability.\n"
        "- Highlight important dietary or lifestyle advice.\n"
        "- Maintain an empathetic and encouraging tone.\n"
        "- MANDATORY: Add this exact safety disclaimer at the end in the target language: "
        "'If your symptoms worsen, please consult your doctor immediately.' (Translate this accurately).\n\n"
        "STRUCTURE:\n"
        "1. Brief condition overview (if provided).\n"
        "2. 💊 Medicines section with details.\n"
        "3. 🩺 Vitals summary (if provided).\n"
        "4. ⚠️ Important advice (if provided).\n"
    )
    
    user_prompt = f"Data to humanize: {json.dumps(data)}"
    
    try:
        response = llm.invoke([
            ("system", system_prompt),
            ("human", user_prompt)
        ])
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"[humanizer] Error: {e}")
        return "I am sorry, I encountered an error while processing your records. Please try again or consult your doctor."
