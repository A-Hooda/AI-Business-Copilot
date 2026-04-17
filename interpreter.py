import json
from config import Config
from llm_manager import LLMManager

class DataInterpreter:
    def __init__(self):
        Config.validate()

    def identify_roles(self, df, custom_persona=""):
        columns = list(df.columns)
        sample_data = df.head(5).to_string() 
        
        # Agnostic summary statistics to help the AI
        stats = df.describe(include='all').to_string()

        prompt = f"""
        {custom_persona}
        You are a Universal Data Intelligence Agent. 
        Your task is to analyze an UNKNOWN dataset and map its headers to FUNCTIONAL ROLES.

        COLUMNS: {columns}
        SAMPLE DATA SNAPSHOT:
        {sample_data}
        
        STATISTICAL SUMMARY:
        {stats[:1000]}

        UNIVERSIAL ROLES:
        - primary_metric: The main numerical value to track (Revenue, Salary, Result).
        - secondary_metric: Other numerical measures.
        - primary_dimension: Chief categorical group (Region, Department, Category).
        - secondary_dimension: Sub-categories.
        - temporal_axis: Date or time columns.
        - identifier: Unique IDs.

        TASK: 
        1. Map every column name to one of the roles above in 'column_roles'.
        2. Identify the 'currency_symbol' (e.g., $, ₹, €) used for metric columns.
        3. Identify the 'currency_code' (e.g., USD, INR, EUR) if possible.
        4. Provide a "business_brief" (one sentence) stating what this dataset represents.
        
        Return ONLY valid JSON with keys: ["column_roles", "currency_symbol", "currency_code", "business_brief"].
        """
        
        content = LLMManager.get_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        content = self.get_clean_json(content)
        return json.loads(content)

    def get_clean_json(self, content):
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return content
