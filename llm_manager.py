import os
import traceback
from groq import Groq
from config import Config

class LLMManager:
    """
    A robust, centralized Multi-Model Waterfall.
    Automatically catches Rate Limit Exceeded (413/429) errors
    and sequentially reroutes the payload to fallback LLMs.
    """
    
    # Ordered fallback list. Start with highest efficiency/speed, fallback to heavy context models.
    FALLBACK_SEQUENCE = [
        "llama-3.1-8b-instant",       
        "llama3-8b-8192",             
        "gemma2-9b-it",              
        "llama-3.3-70b-versatile",    
        "mixtral-8x7b-32768"             
    ]

    @staticmethod
    def get_completion(messages, temperature=0.5):
        Config.validate()
        client = Groq(api_key=Config.GROQ_API_KEY)
        
        # Pull default preferred model, move it to the front of the line
        preferred_model = getattr(Config, "SUPPORTED_MODEL", "llama-3.1-8b-instant")
        models_to_try = [preferred_model] + [m for m in LLMManager.FALLBACK_SEQUENCE if m != preferred_model]

        last_error = None
        
        for model in models_to_try:
            try:
                # Debug log to terminal to let the user visually trace the waterfall behavior
                print(f"--- [LLM Manager] Attempting routing via model: {model} ---")
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
                
                print(f"--- [LLM Manager] SUCCESS! Payloaded accepted by {model} ---")
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e).lower()
                last_error = e
                print(f"--- [LLM Manager] WARNING: {model} failed! Reason: {str(e)[:100]}... ---")
                
                # Check if it is a Rate Limit/Token ceiling error (413, 429)
                if "rate limit" in error_str or "request too large" in error_str or "rate_limit_exceeded" in error_str or "413" in error_str or "429" in error_str or "model_decommissioned" in error_str:
                    print(f"--- [LLM Manager] Initiating Waterfall Fallback to next available model... ---")
                    continue # Let the loop try the next model
                else:
                    # If it's an API key error or internal parsing issue, we crash gracefully.
                    raise e
                    
        # If we exhausted all fallback models
        raise Exception(f"CRITICAL: All {len(models_to_try)} models in the Waterfall failed due to Token/Rate limit ceilings! Final error: {str(last_error)}")
