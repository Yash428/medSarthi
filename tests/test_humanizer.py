import json
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.humanizer import humanize_medical_response

def test_humanizer():
    # Example data from user request
    sample_data = {
        "condition": "Cold and Fever",
        "vitals": {
            "temperature": 102,
            "blood_pressure": "120/80",
            "pulse": 112
        },
        "medicines": [
            {
                "name": "Paracetamol",
                "timing": "1-0-1",
                "food": "After Food",
                "duration": "10 days"
            },
            {
                "name": "Montelukast",
                "timing": "1-0-1",
                "food": "After Food",
                "duration": "10 days"
            }
        ],
        "dietary_advice": "Avoid cold food",
        "doctor_notes": "Patient has fever and cold"
    }

    print("--- Testing Humanizer (Gujarati) ---")
    gu_response = humanize_medical_response(sample_data, language_code="gu")
    print(gu_response)
    print("\n" + "="*50 + "\n")

    print("--- Testing Humanizer (English) ---")
    en_response = humanize_medical_response(sample_data, language_code="en")
    print(en_response)

if __name__ == "__main__":
    test_humanizer()
