import pandas as pd
from datetime import datetime


class HRExpert:
    def preprocess(self, df, roles):
        """
        Specialized HR analytics logic.
        """
        hire_date_col = None
        salary_col = None

        # 1. Identify key columns from roles
        for col, role in roles.items():
            if role == 'temporal_axis':
                hire_date_col = col
            if role == 'primary_metric':
                salary_col = col

        # 2. Tenure Calculation (Years since Hire Date)
        if hire_date_col and hire_date_col in df.columns:
            print(f"--- [HR Expert] Calculating Employee Tenure from {hire_date_col} ---")
            df[hire_date_col] = pd.to_datetime(df[hire_date_col], errors='coerce')
            current_date = datetime.now()
            df['Tenure_Years'] = df[hire_date_col].apply(
                lambda x: (current_date - x).days / 365.25 if pd.notnull(x) else 0
            )

        # 3. Salary Parity Score
        if salary_col and salary_col in df.columns:
            mean_sal = df[salary_col].mean()
            if mean_sal and mean_sal != 0:
                df['Salary_vs_Avg'] = df[salary_col] / mean_sal

        return df

    @staticmethod
    def get_consultant_prompt():
        return """
        Act as a Chief Human Resources Officer (CHRO) and Workforce Strategist.
        Focus on:
        - Employee longevity (Tenure) and Retention.
        - Compensation fairness (Salary vs Average).
        - Identifying high-cost departments or flight risks.
        """
