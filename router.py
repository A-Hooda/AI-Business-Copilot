import json
from config import Config
from llm_manager import LLMManager

class ExpertRouter:
    def __init__(self):
        Config.validate()

    def determine_domain(self, df):
        """
        Analyze columns and sample data to determine the dataset's domain.
        """
        columns = list(df.columns)
        sample = df.head(3).to_string()
        
        prompt = f"""
        You are a Data Classifier. Analyze this dataset and determine its domain.
        
        COLUMNS: {columns}
        SAMPLE: {sample}
        
        Choose the best fit from: 'finance', 'hr', 'general'.
        - 'finance': If it contains sales, revenue, profit, tax, or balance sheets.
        - 'hr': If it contains employees, salaries, hiring, performance, or departments.
        - 'general': Anything else.

        Return ONLY a JSON object: {{"domain": "choice"}}
        """
        
        response_text = LLMManager.get_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        try:
            content = response_text.strip()
            # Basic cleanup if AI adds markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content).get("domain", "general")
        except:
            return "general"
