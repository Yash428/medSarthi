import requests
import os
from src.config import settings

class STTService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.sarvam.ai/speech-to-text"
        print("[STTService] Initialized with Sarvam AI.")

    def transcribe(self, audio_path: str, language: str = None) -> str:
        """
        Transcribes the audio file using Sarvam AI Saaras v3.
        """
        if not self.api_key:
            print("[STTService] Error: Missing SARWAM_API_KEY")
            return "Error: STT not configured."

        # Map language codes to Sarvam format (e.g. 'hi' -> 'hi-IN')
        lang_map = {
            "hi": "hi-IN",
            "gu": "gu-IN",
            "en": "en-IN"
        }
        sarvam_lang = lang_map.get(language, "en-IN")

        try:
            with open(audio_path, "rb") as f:
                files = {
                    "file": (os.path.basename(audio_path), f, "audio/webm")
                }
                data = {
                    "model": "saaras:v3",
                    "language_code": "unknown",
                    "mode": "transcribe"
                }
                headers = {
                    "api-subscription-key": self.api_key
                }
                
                print(f"[STTService] Requesting Sarvam STT (Model: saaras:v3, Mode: transcribe, Lang: {sarvam_lang})")
                response = requests.post(self.url, files=files, data=data, headers=headers)
                
                if response.status_code != 200:
                    print(f"[STTService] Error {response.status_code}: {response.text}")
                    return f"Error: STT API returned {response.status_code}"
                
                result = response.json()
                print(f"[STTService] Raw Result: {result}")
                
                transcript = result.get("transcript", "")
                transcript = transcript.strip()
                
                print(f"[STTService] Final Transcript: '{transcript}'")
                return transcript
        except Exception as e:
            print(f"[STTService] Sarvam STT error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                 print(f"[STTService] Response: {e.response.text}")
            return ""

# Singleton instance
stt_engine = STTService(api_key=settings.SARWAM_API_KEY)
