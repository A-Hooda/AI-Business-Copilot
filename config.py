import os
from dotenv import load_dotenv

# Initialize project configuration
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    SUPPORTED_MODEL = "mixtral-8x7b-32768"

    @staticmethod
    def validate():
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in .env file.")
