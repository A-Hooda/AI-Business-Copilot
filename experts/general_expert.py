import pandas as pd


class GeneralExpert:
    def preprocess(self, df, roles):
        """
        Universal preprocessing for any unknown dataset.
        """
        inv_map = {role: col for col, role in roles.items()}
        target = inv_map.get('primary_metric')
        dim = inv_map.get('primary_dimension')

        if target and dim:
            print(f"--- [General Expert] Performing Aggregation on {target} by {dim} ---")
            df[f'total_{target}_by_{dim}'] = df.groupby(dim)[target].transform('sum')
            df[f'avg_{target}_by_{dim}'] = df.groupby(dim)[target].transform('mean')

        return df

    @staticmethod
    def get_consultant_prompt():
        return """
        Act as a Lead Data Scientist and Universal Business Intelligence Agent.
        Focus on:
        - Broad trends and data distribution.
        - Identifying outliers and hidden relationship patterns.
        - Translating data roles into business value.
        """
