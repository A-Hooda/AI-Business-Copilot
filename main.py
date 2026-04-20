import sys
import pandas as pd
from ingester import GenericIngester
from interpreter import DataInterpreter
from adapter import DataAdapter
from router import ExpertRouter
from predictor import UniversalPredictor
from visualizer import AgnosticVisualizer
from advisor import ExpertAdvisor

# Import Experts
from experts.finance_expert import FinanceExpert
from experts.hr_expert import HRExpert
from experts.general_expert import GeneralExpert

def main():
    # 1. Flexible Ingestion
    file_path = sys.argv[1] if len(sys.argv) > 1 else "sales_data.csv"
    
    try:
        df = GenericIngester.load_data(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # 2. Domain Routing (Front Desk)
    print("\n--- [Copilot] Routing to Specialist... ---")
    router = ExpertRouter()
    domain = router.determine_domain(df)
    
    # Select Expert Strategy
    if domain == 'finance':
        expert = FinanceExpert()
        title = "FINANCIAL ANALYST"
    elif domain == 'hr':
        expert = HRExpert()
        title = "HR ANALYST"
    else:
        expert = GeneralExpert()
        title = "GENERAL ANALYST"
        
    print(f"--- [Copilot] Handing over to: {title} ---")

    # 3. Initial Mapping & Cleaning (The Brain)
    interpreter = DataInterpreter()
    mapping = interpreter.identify_roles(df)
    roles = mapping.get("column_roles", {})
    df, _ = DataAdapter.auto_clean(df, roles)


    # 4. Expert Preprocessing
    df = expert.preprocess(df, roles)
    
    # 5. Diagnostic Intelligence (The "Why")
    print("\n--- [Copilot] Running ML Diagnostics to find performance drivers... ---")
    
    # Scale for Neural Network stability
    df_scaled = DataAdapter.scale_for_ml(df, roles)
    prediction_payload = UniversalPredictor.get_performance_drivers(df_scaled, roles)
    if prediction_payload:
        drivers, y_test, y_pred, mae, r2 = prediction_payload
    else:
        drivers, y_test, y_pred, mae, r2 = {}, None, None, 0.0, 0.0
    
    # 6. Automated Visualization (The Charts)
    print("\n--- [Copilot] Generating Diagnostic Visualizations... ---")
    forecast_df = UniversalPredictor.generate_forecast(df, roles)
    AgnosticVisualizer.create_reports(df, roles, drivers, y_test, y_pred, forecast_df=forecast_df)
    
    # 7. Strategic Advice (The Recommendations)
    print("\n--- [Copilot] Generating Executive Strategy Report... ---")
    advisor = ExpertAdvisor()
    persona = expert.get_consultant_prompt()
    
    # Re-run role mapping with persona for deeper context
    final_mapping = interpreter.identify_roles(df, custom_persona=persona)
    insight = final_mapping.get("business_brief", "No insight generated.")
    
    strategy = advisor.generate_strategy(df, roles, drivers, domain, insight)
    
    # 8. Results & File Export
    with open("EXECUTIVE_STRATEGY.md", "w", encoding='utf-8') as f:
        f.write("# EXECUTIVE STRATEGY REPORT\n\n")
        f.write(strategy)
        
    print(f"\n[{title} ANALYSIS COMPLETE]")
    print(f"KEY INSIGHT: {insight}")
    print("\n--- [SUCCESS] Strategic Report saved: EXECUTIVE_STRATEGY.md")
    print("--- [SUCCESS] Data Visuals saved: /reports")

if __name__ == "__main__":
    main()
