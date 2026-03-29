import easyocr
import os
import json
import logging
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize EasyOCR reader
# First usage will download the model weights (languages: english)
try:
    reader = easyocr.Reader(['en'])
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR: {e}")
    reader = None

def extract_text_from_file(file_path: str) -> str:
    """Extract raw text from an image using EasyOCR."""
    if reader is None:
        return "OCR engine not initialized."
    
    try:
        # EasyOCR can handle images directly. 
        # For PDFs, we'll need to convert to image first (handled in the router ideally)
        results = reader.readtext(file_path, detail=0)
        return " ".join(results)
    except Exception as e:
        logger.error(f"OCR Extraction failed: {e}")
        return f"Error during OCR: {str(e)}"

def parse_medical_data_with_llm(raw_text: str) -> List[Dict[str, str]]:
    """Use Groq/LLM to transform raw OCR text into structured medical metrics."""
    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not found. Returning empty list.")
        return []

    llm = ChatGroq(
        temperature=0,
        groq_api_key=settings.GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a medical data extraction assistant. "
            "Your task is to extract lab test metrics and their values from raw OCR text of a medical report. "
            "Return ONLY a JSON array of objects with 'key' and 'value' fields. "
            "Keys should be the test name (e.g., 'Hemoglobin', 'WBC Count'). "
            "Values should be the numeric result with units if available. "
            "If no metrics are found, return an empty array []. "
            "Strictly return ONLY the JSON."
        )),
        ("human", "Raw OCR Text: {text}")
    ])

    try:
        chain = prompt | llm
        response = chain.invoke({"text": raw_text})
        
        # Extract JSON from response
        content = response.content.strip()
        # Handle cases where LLM might wrap JSON in markdown blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
            
        return json.loads(content)
    except Exception as e:
        logger.error(f"LLM Parsing failed: {e}")
        # Fallback: maybe just return some basic regex-based extraction if LLM fails?
        # For now, return empty to let user manual edit.
        return []
