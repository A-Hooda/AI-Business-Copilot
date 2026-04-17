import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class DataAdapter:
    @staticmethod
    def auto_clean(df, roles):
        """
        Phase 1 - Cleaning only (NO scaling).
        Scaling before expert preprocessing breaks domain-specific % calculations
        (e.g. tax = revenue * 0.18 on a z-score is meaningless).
        Scaling for the ML predictor happens separately inside get_performance_drivers.
        """
        print(f"--- [Adapter] Cleaning {len(df)} rows... ---")
        df = df.copy()

        # 0. Robustness for Enterprise Scale
        # Drop strict duplicates early to save memory/tokens
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            print(f"--- [Adapter] Dropped {initial_rows - len(df)} duplicate records. ---")

        # Row Sampling Ceiling (1 Million Rows)
        # Prevents kernel crashes on memory-constrained environments while preserving distribution
        if len(df) > 1000000:
            print(f"--- [Adapter] Dataset too large ({len(df)} rows). Downsampling to 1M rows for stability. ---")
            df = df.sample(n=1000000, random_state=42)

        # 0. Proactive Currency & String Metric Coercion
        # Force designated metric columns to float if they contain nasty strings (e.g., "$1,200", "1,000.00")
        for c, r in roles.items():
            if r in ['primary_metric', 'secondary_metric'] and c in df.columns:
                if df[c].dtype == 'object':
                    # Strip any non-numeric sequence EXCEPT . and - indicating decimals/negatives
                    df[c] = df[c].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # 1. Handle Missing Values
        # Only apply numerical imputation to columns that are actually numeric
        num_cols = [
            c for c, r in roles.items() 
            if r in ['primary_metric', 'secondary_metric'] 
            and c in df.columns 
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        
        if num_cols:
            # Use Pandas fillna instead of sklearn SimpleImputer to prevent dropping completely empty columns (which causes key mismatch scaling bugs)
            for c in num_cols:
                median_val = df[c].median()
                if pd.isna(median_val):
                    median_val = 0.0 # Fallback if entirely NaN
                df[c] = df[c].fillna(median_val)

        # Treat everything else identified as a metric but not numeric as a categorical dimension for cleaning
        cat_cols = [
            c for c, r in roles.items() 
            if (r in ['primary_dimension', 'secondary_dimension'] or (r in ['primary_metric', 'secondary_metric'] and not pd.api.types.is_numeric_dtype(df[c])))
            and c in df.columns
        ]
        if cat_cols:
            # Use Pandas for robust categorical imputation
            for c in cat_cols:
                mode_series = df[c].mode()
                mode_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                if pd.isna(mode_val):
                    mode_val = "Unknown"
                df[c] = df[c].fillna(mode_val).astype(str)

        # 2. Date Standardization
        time_col = next((c for c, r in roles.items() if r == 'temporal_axis' and c in df.columns), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

        print(f"--- [Adapter] Cleaning complete. Shape: {df.shape} ---")
        return df

    @staticmethod
    def scale_for_ml(df, roles):
        """
        Phase 2 - StandardScaler for numeric columns ONLY.
        Call this AFTER expert preprocessing if you need scaled features for DL.
        """
        num_cols = [
            c for c, r in roles.items() 
            if r in ['primary_metric', 'secondary_metric'] 
            and c in df.columns 
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if num_cols:
            scaler = StandardScaler()
            df = df.copy()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            print(f"--- [Adapter] Scaled {len(num_cols)} numerical features for ML. ---")
        return df

    @staticmethod
    def add_agnostic_metrics(df, mapping):
        """
        Calculate aggregates based on functional roles rather than hardcoded names.
        """
        inv_map = {role: col for col, role in mapping.items()}
        target = inv_map.get('primary_metric')
        dim = inv_map.get('primary_dimension')

        if target and dim and target in df.columns and dim in df.columns:
            print(f"--- [Adapter] Aggregating {target} by {dim} ---")
            df[f'total_{target}_by_{dim}'] = df.groupby(dim)[target].transform('sum')
            df[f'avg_{target}_by_{dim}'] = df.groupby(dim)[target].transform('mean')

        return df

    @staticmethod
    def feature_engineer(df, mapping):
        """
        Extract temporal features based on functional roles.
        """
        for col, role in mapping.items():
            if role == 'temporal_axis' and col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    prefix = col.lower()
                    df[f'{prefix}_month'] = df[col].dt.month
                    df[f'{prefix}_weekday'] = df[col].dt.day_name()
        return df
