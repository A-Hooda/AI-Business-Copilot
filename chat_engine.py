from config import Config
from computation_engine import DataComputer
from llm_manager import LLMManager

class DataChatter:
    def __init__(self):
        Config.validate()

    def ask(self, df, mapping, question, currency_symbol="$"):
        """
        Answers a user question based on the provided dataframe and AI-identified roles.
        """
        # 1. Prepare Data Context
        columns = list(df.columns)
        head = df.head(5).to_string()
        stats = df.describe(include='all').to_string()
        
        # Identify the main KPI for the AI to focus on
        target = next((c for c, r in mapping.items() if r == 'primary_metric'), 'Metric')
        
        # EXTRACT EXACT MATH COMPUTATIONS!
        # Because LLMs are bad at math, we explicitly run Pandas math and inject it 
        # as a hard dictionary so the model knows the exact answers (like Total Sales in North).
        computed_aggregates = DataComputer.compute_all(df, mapping)
        
        prompt = f"""
        You are a Conversational BI Assistant. 
        You have access to a dataset with these columns: {columns}
        The currency used in this data is: {currency_symbol}
        
        DATA SUMMARY:
        {stats[:1500]}
        
        SAMPLE DATA:
        {head[:1000]}
        
        PRE-CALCULATED MATHEMATICAL AGGREGATES (USE THESE FOR EXACT ANSWERS):
        {str(computed_aggregates)[:4000]}

        USER QUESTION: "{question}"
        
        INSTRUCTIONS:
        - Analyze the data context and the PRE-CALCULATED AGGREGATES to answer the user's specific question.
        - You MUST use the exact numbers from the 'PRE-CALCULATED MATHEMATICAL AGGREGATES' JSON instead of calculating yourself.
        - If the user asks for exact regional totals, sales, profit, or percentage drops, retrieve them directly from the computed JSON dict.
        - CAUSAL INFERENCE MANDATE: Whenever you present a numerical computation, you MUST provide a plausible, real-world business reason for WHY that trend exists. Do not just state the data; infer target demographics, macro-economic conditions, or supply/demand factors (e.g., if laptop sales are high in a region, assume there are IT hubs or university students).
        - Be authoritative, accountable, and sound like a Senior Director of Strategy when reasoning. 
        - Keep your answer concise: 3-5 sentences max.
        - If you can't answer from the provided data or aggregates, say so politely.
        """
        
        response_text = LLMManager.get_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        
        return response_text
