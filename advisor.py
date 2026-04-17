from config import Config
from computation_engine import DataComputer
from llm_manager import LLMManager

class ExpertAdvisor:
    def __init__(self):
        Config.validate()

    def generate_strategy(self, df, roles, drivers, domain, insight, currency_symbol="$"):
        """
        Generates a comprehensive Markdown report with diagnostic reasoning and strategy.
        """
        # 1. Identify "Low Performers" for the AI to reason about
        target = next((c for c, r in roles.items() if r == 'primary_metric'), 'Metric')
        dim = next((c for c, r in roles.items() if r == 'primary_dimension'), 'Category')
        
        # Sort by worst performers
        worst_segments = df.groupby(dim)[target].mean().sort_values(ascending=True).head(3).to_dict()
        top_drivers = list(drivers.keys())[:3] if drivers else []
        
        # EXTRACT EXACT MATH COMPUTATIONS
        computed_aggregates = DataComputer.compute_all(df, roles)

        prompt = f"""
        You are a Senior Business Consultant specializing in {domain}.
        
        SITUATION ANALYSIS:
        - Primary Goal: Optimize {target}.
        - Currency: {currency_symbol}
        - Key Performance Drivers: {top_drivers}
        - Current Baseline Insight: {insight}
        
        PROBLEM AREAS (Lowest 3 segments by {target} in {currency_symbol}):
        {worst_segments}
        
        PRE-CALCULATED MATHEMATICAL AGGREGATES:
        {str(computed_aggregates)[:8000]}

        TASK:
        Generate a professional Executive Strategy Report in Markdown format.
        Include:
        1. ## Diagnostic & Causal Summary: Why are these 3 segments performing poorly? Use exact numbers from the aggregates. CRITICAL: Provide explicit, real-world hypotheses (demographics, seasonal demand, infrastructural gaps) for WHY these numbers look the way they do! Be accountable and authoritative.
        2. ## Root Cause Analysis: How do the key drivers ({top_drivers}) explain this? Establish direct theoretical links.
        3. ## Actionable Recommendations: Give 3 specific, domain-expert steps aiming to resolve the causal issues you just hypothesized.
        
        Return ONLY the Markdown content. Do not include any dialogue.
        """
        
        response_text = LLMManager.get_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response_text