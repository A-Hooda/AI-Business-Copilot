import pandas as pd
import numpy as np


class FinanceExpert:
    def preprocess(self, df, roles):
        """
        Specialized financial cleaning and deep feature engineering for Neural Networks.
        """
        print("--- [Finance Expert] Deep Feature Engineering for DL ---")

        metric = None
        for col, role in roles.items():
            if role == 'primary_metric':
                metric = col
                break

        if metric:
            print(f"--- [Finance Expert] Calculating Margins and Taxes for {metric} ---")
            # 1. Estimate Profit (15%) and Taxation (18%)
            df['Estimated_Tax'] = df[metric] * 0.18
            df['Net_Profit_Estimate'] = (df[metric] * 0.15) - (df[metric] * 0.05)

            # 2. Performance Indicator: Is this high/low revenue?
            benchmark = df[metric].median()
            df['Rev_Performance'] = df[metric].apply(
                lambda x: 'High' if x > benchmark else 'Low'
            )

            # 3. Advanced Numerical Transformation (Log Scale for skewed sales)
            print(f"--- [Finance Expert] Applying Log-Transform to {metric} ---")
            df[metric] = np.log1p(df[metric].clip(lower=0))

        return df

    @staticmethod
    def get_consultant_prompt():
        return """
        Act as a Senior CFO and Financial Forensic Accountant.
        Focus on:
        - Profit Margins and Fiscal health.
        - Tax liabilities and Cash flow trends.
        - Identifying high-revenue but low-efficiency segments.
        """
