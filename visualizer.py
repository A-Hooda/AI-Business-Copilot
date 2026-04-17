import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend (safe for server use)
import matplotlib.pyplot as plt
import seaborn as sns
import os


class AgnosticVisualizer:
    @staticmethod
    def create_reports(df, roles, drivers, y_test=None, y_pred=None, forecast_df=None, output_dir='reports'):
        """
        Generates automated diagnostic charts based on roles and drivers.
        Renders precisely 9 required graphs mapping to Sections 2-5 of the dashboard.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            for fd in os.listdir(output_dir):
                if fd.endswith('.png'):
                    try:
                        os.remove(os.path.join(output_dir, fd))
                    except Exception:
                        pass

        # Cyber-Aura Theme colors
        plt.rcParams.update({
            'figure.facecolor': '#050814',
            'axes.facecolor':   '#0d1126',
            'axes.edgecolor':   '#2a1c4d',
            'axes.labelcolor':  '#a7b5c9',
            'xtick.color':      '#a7b5c9',
            'ytick.color':      '#a7b5c9',
            'text.color':       '#f2f7fb',
            'grid.color':       '#2a1c4d',
            'grid.alpha':       0.5,
        })

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        target = next((c for c, r in roles.items() if r == 'primary_metric' and c in df.columns), None)
        dim = next((c for c, r in roles.items() if r == 'primary_dimension' and c in df.columns), None)
        time_col = next((c for c, r in roles.items() if r == 'temporal_axis' and c in df.columns), None)
        
        # Fallbacks to guarantee robust plots
        if not target and numeric_cols: target = numeric_cols[0]
        if not dim and cat_cols: dim = df[cat_cols].nunique().idxmin()

        # ---------------------------------------------------------
        # SECTION 2: Exploratory Analysis (Histogram, Box, Heatmap)
        # ---------------------------------------------------------
        
        # 1. Histogram (dist.png)
        print("--- [Visualizer] Generating Histogram (dist.png) ---")
        fig, ax = plt.subplots(figsize=(10, 6))
        if target:
            sns.histplot(df[target].dropna(), kde=True, ax=ax, color='#00ff88', bins=20)
            ax.set_title(f'{target} Distribution', fontsize=14, fontweight='bold', color='#00ff88', pad=15)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'dist.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        # 2. Box Plot / Bar Chart (box.png)
        print("--- [Visualizer] Generating Box/Bar Plot (box.png) ---")
        fig, ax = plt.subplots(figsize=(10, 6))
        if target and dim:
            if df[dim].nunique() < 15:
                # Use Boxplot across categories
                sns.boxplot(data=df, x=dim, y=target, hue=dim, palette='BuPu', legend=False, ax=ax)
                ax.set_title(f'{target} Variance by {dim}', fontsize=14, fontweight='bold', color='#00e5ff', pad=15)
            else:
                # Use category aggregation Bar chart
                top_segments = df.groupby(dim)[target].sum().sort_values(ascending=False).head(10)
                sns.barplot(x=top_segments.values, y=top_segments.index, palette='BuPu', ax=ax)
                ax.set_title(f'Top {target} by {dim}', fontsize=14, fontweight='bold', color='#00e5ff', pad=15)
        else:
            if target: sns.boxplot(y=df[target], color='#00e5ff', ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'box.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        # 3. Correlation Heatmap (correlation.png)
        print("--- [Visualizer] Generating Correlation Heatmap (correlation.png) ---")
        fig, ax = plt.subplots(figsize=(10, 8))
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='Purples', ax=ax, fmt=".2f", linewidths=.5, cbar_kws={"shrink": .8})
            ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', color='#00e5ff', pad=15)
            plt.xticks(rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'Requires >1 Numeric Columns', fontsize=16, color='#8aad99', ha='center', va='center')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'correlation.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        # ---------------------------------------------------------
        # SECTION 3: Trends (Line chart)
        # ---------------------------------------------------------
        
        # 4. Line Chart (trend.png)
        print("--- [Visualizer] Generating Line Chart (trend.png) ---")
        fig, ax = plt.subplots(figsize=(10, 6))
        if time_col and target:
            df_time = df.copy()
            df_time[time_col] = pd.to_datetime(df_time[time_col], errors='coerce')
            trend = df_time.set_index(time_col)[target].resample('ME').mean() if len(df_time)>100 else df_time.set_index(time_col)[target]
            sns.lineplot(x=trend.index, y=trend.values, ax=ax, color='#00e5ff', marker='o')
            ax.set_title(f'{target} Trend Over Time', fontsize=14, fontweight='bold', color='#00e5ff', pad=15)
        elif dim and target:
            ordered_trend = df.groupby(dim)[target].mean()
            sns.lineplot(x=ordered_trend.index, y=ordered_trend.values, ax=ax, color='#00e5ff', marker='o')
            ax.set_title(f'{target} Cross-Segment Trend', fontsize=14, fontweight='bold', color='#00e5ff', pad=15)
        else:
            ax.text(0.5, 0.5, 'Requires Temporal/Category', fontsize=16, color='#8aad99', ha='center')
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'trend.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        # ---------------------------------------------------------
        # SECTION 4: Scalable ML Evaluation
        # ---------------------------------------------------------

        # 5. Scatter Plot (scatter.png)
        print("--- [Visualizer] Generating Scatter Plot (scatter.png) ---")
        fig, ax = plt.subplots(figsize=(10, 6))
        if target and len(numeric_cols) > 1:
            correlations = df[numeric_cols].corr()[target].abs().sort_values(ascending=False).drop(target)
            if not correlations.empty:
                top_feature = correlations.index[0]
                sns.scatterplot(data=df, x=top_feature, y=target, hue=dim if dim else None, palette='cool', alpha=0.7, ax=ax)
                ax.set_title(f'{target} vs {top_feature}', fontsize=14, fontweight='bold', color='#00e5ff', pad=15)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'scatter.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        # 6. Prediction vs Actual Plot (pred_vs_actual.png)
        print("--- [Visualizer] Generating Prediction vs Actual (pred_vs_actual.png) ---")
        fig, ax = plt.subplots(figsize=(10, 6))
        if y_test is not None and y_pred is not None:
            sns.scatterplot(x=y_test.flatten(), y=y_pred.flatten(), color='#00e5ff', alpha=0.6, ax=ax)
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], color='#b000ff', linestyle='--', linewidth=2)
            ax.set_title('Prediction vs Actual', fontsize=14, fontweight='bold', color='#00e5ff', pad=15)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
        else:
            ax.text(0.5, 0.5, 'No Predictions Available', ha='center', color='#8aad99')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'pred_vs_actual.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        # 7. Residual Plot (residual.png)
        print("--- [Visualizer] Generating Residual Plot (residual.png) ---")
        fig, ax = plt.subplots(figsize=(10, 6))
        if y_test is not None and y_pred is not None:
            residuals = y_test.flatten() - y_pred.flatten()
            sns.scatterplot(x=y_pred.flatten(), y=residuals, color='#00ff88', alpha=0.6, ax=ax)
            ax.axhline(0, color='#b000ff', linestyle='--', linewidth=2)
            ax.set_title('Residual Error Distribution', fontsize=14, fontweight='bold', color='#00ff88', pad=15)
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Errors (Actual - Predicted)')
        else:
            ax.text(0.5, 0.5, 'No Residuals Available', ha='center', color='#8aad99')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'residual.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        # ---------------------------------------------------------
        # SECTION 5: Additions 
        # ---------------------------------------------------------
        
        # 8. Pie Chart (pie.png)
        print("--- [Visualizer] Generating Pie Chart (pie.png) ---")
        fig, ax = plt.subplots(figsize=(8, 8))
        if dim and target:
            composition = df.groupby(dim)[target].sum().sort_values(ascending=False).head(5)
            colors = sns.color_palette('cool', len(composition))
            ax.pie(composition, labels=composition.index, autopct='%1.1f%%', colors=colors, textprops={'color':"w"})
            ax.set_title(f'Top {dim} Proportion', fontsize=14, fontweight='bold', color='#00e5ff', pad=10)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'pie.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        # 9. Feature Interaction Plot (interaction.png)
        print("--- [Visualizer] Generating Feature Interaction / SHAP Importance (interaction.png) ---")
        fig, ax = plt.subplots(figsize=(10, 6))
        if drivers:
            driver_df = pd.DataFrame(list(drivers.items()), columns=['Feature', 'Importance']).head(10)
            driver_df = driver_df.sort_values(by='Importance', ascending=True)
            sns.barplot(data=driver_df, x='Importance', y='Feature', hue='Feature', palette='Reds_r', legend=False, ax=ax)
            ax.set_title('Feature Importance / SHAP Aggregation', fontsize=14, fontweight='bold', color='#ff4d4d', pad=15)
            ax.set_xlabel('Model Weight Impact')
        else:
            ax.text(0.5, 0.5, 'Insufficient Model Drivers', ha='center', color='#8aad99')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'interaction.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        print(f"--- [Visualizer] All 9 charts generated successfully to /{output_dir} ---")

        # 10. Forecasting Graph (`forecast.png`)
        if forecast_df is not None:
            print("--- [Visualizer] Generating Forecasting Horizon Plot ---")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot historical (non-null 'Actual')
            hist_df = forecast_df.dropna(subset=['Actual'])
            ax.plot(hist_df['Date'], hist_df['Actual'], color='#00e5ff', label='Historical Actual', linewidth=2)
            
            # Plot prediction line
            ax.plot(forecast_df['Date'], forecast_df['Forecast'], color='#ff007f', linestyle='--', label='ML Forecast', linewidth=2)
            
            ax.set_title(f"Future Projection Horizon ({target})", fontsize=14, fontweight='bold', color='#f2f7fb', pad=15)
            ax.set_xlabel("Time Axis")
            ax.set_ylabel(target if target else "Metric")
            
            ax.legend(facecolor='#0d1126')
            
            # Rotate x-axis labels if it gets too crowded
            plt.xticks(rotation=45)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'forecast.png'), dpi=120, bbox_inches='tight')
            plt.close(fig)
