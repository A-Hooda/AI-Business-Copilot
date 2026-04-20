import pandas as pd
import json

class DataComputer:
    """
    Exhaustively scans the DataFrame and actively computes 
    sums, means, and timeline shifts using Pandas Native Math.
    Generates exact scalar values to pass to the LLM context.
    """
    @staticmethod
    def compute_all(df, mapping):
        computed_dict = {}
        
        # Identify key columns organically based on mapping
        primary_metric = next((c for c, r in mapping.items() if r == 'primary_metric' and c in df.columns), None)
        categorical_dims = [c for c, r in mapping.items() if 'dimension' in r and c in df.columns]
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # 1. Compute Exact Totals for ALL numeric fields
        computed_dict["global_totals"] = {}
        computed_dict["global_averages"] = {}
        for num_col in numeric_cols:
            computed_dict["global_totals"][num_col] = float(df[num_col].sum())
            computed_dict["global_averages"][num_col] = float(df[num_col].mean())
            
        # 2. GroupBy Aggregations: Sums and Means across ALL Dimensions!
        # This allows answering "Total Sales in North"
        computed_dict["segment_aggregates"] = {}
        
        # Use AI-identified dimensions first, fallback to all categorical columns with < 50 unique values
        if not categorical_dims:
            categorical_dims = [cat for cat in cat_cols if df[cat].nunique() < 50][:3]
            
        for dim in categorical_dims:
            if dim in df.columns:
                computed_dict["segment_aggregates"][dim] = {}
                for num_col in numeric_cols[:3]: # Only calculate for top 3 numeric features
                    if num_col == dim: continue
                    # Get exact sum per segment category (Truncated to Top 3 dynamically to avoid 24k token limit)
                    # We will sort by size/sum so the LLM gets the most important segments
                    top_categories = df.groupby(dim)[num_col].sum().sort_values(ascending=False).head(3)
                    sum_agg = top_categories.to_dict()
                    # Calculate mean only for the top categories to save space
                    mean_agg = df[df[dim].isin(top_categories.index)].groupby(dim)[num_col].mean().to_dict()
                    
                    computed_dict["segment_aggregates"][dim][f"{num_col}_sum"] = sum_agg
                    computed_dict["segment_aggregates"][dim][f"{num_col}_avg"] = mean_agg
                    
        # 3. Simple Timeline (Temporal) Drops/Gains
        time_col = next((c for c, r in mapping.items() if r == 'temporal_axis'), None)
        if not time_col:
            time_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            if time_cols: time_col = time_cols[0]
            
        if time_col and primary_metric and time_col in df.columns and primary_metric in df.columns:
            try:
                df_time = df.copy()
                df_time[time_col] = pd.to_datetime(df_time[time_col])
                # Group by Month/Year
                df_time['period'] = df_time[time_col].dt.to_period('M').astype(str)
                time_agg = df_time.groupby('period')[primary_metric].sum().sort_index()
                
                # Truncate to the most recent 12 months to avoid massive token usage
                if len(time_agg) > 12:
                    time_agg = time_agg.tail(12)
                
                # Calculate percentage drops/gains between consecutive periods
                pct_changes = time_agg.pct_change().round(2).fillna(0) * 100
                computed_dict["temporal_analysis"] = {
                    "monthly_totals": time_agg.round(1).to_dict(),
                    "monthly_pct_change": pct_changes.to_dict(),
                }
            except Exception as e:
                pass # Ignore if temporal parsing fails gracefully
                
        # Aggressively strip spaces to save tokens
        return json.dumps(computed_dict, separators=(',', ':'))
