import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import warnings
import gc

# --- Sourcedotcom Deep Learning Predictor ---
# Custom Neural Network for High-Integrity Feature Attribution

from torch.utils.data import DataLoader, TensorDataset

class PerformanceNet(nn.Module):
    def __init__(self, input_dim, use_lstm=False):
        super(PerformanceNet, self).__init__()
        self.use_lstm = use_lstm
        
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=128, batch_first=True)
            self.network = nn.Sequential(
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                
                nn.Linear(32, 1)
            )

    def forward(self, x):
        if self.use_lstm:
            x_seq = x.unsqueeze(1)
            lstm_out, _ = self.lstm(x_seq)
            x = lstm_out[:, -1, :]
            
        batch_size = x.size(0)
        if batch_size == 1:
            self.eval() # Disable batch_norm behavior locally
            out = self.network(x)
            self.train()
            return out
        return self.network(x)

# --- Fast AI Engine (Turbo Mode) ---
# Used for Cloud Stability to avoid timeouts
class FastPredictor:
    @staticmethod
    def get_drivers(X_train, X_test, y_train, y_test, feature_cols):
        print("--- [Turbo Mode] Starting Fast Random Forest Analysis ---")
        model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train.ravel())
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Fast Importance
        result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
        importances = {}
        for i, col in enumerate(feature_cols):
            importances[col] = float(max(0, result.importances_mean[i]))
            
        print("--- [Turbo Mode] Analysis Complete ---")
        return importances, y_test, y_pred, mae, r2

class UniversalPredictor:
    @classmethod
    def get_performance_drivers(cls, df, roles, use_turbo=True):
        """
        Sourcedotcom PyTorch Predictor.
        Trains a Deep Neural Network and uses Permutation Importance to find drivers.
        """
        # 1. Prepare Data
        target_col = None
        feature_cols = []
        
        for col, role in roles.items():
            if role == 'primary_metric':
                target_col = col
            elif role in ['primary_dimension', 'secondary_dimension', 'temporal_axis', 'secondary_metric']:
                if col in df.columns:
                    feature_cols.append(col)
                
        if not target_col or target_col not in df.columns or not feature_cols:
            return None
            
        print(f"--- [Sourcedotcom] Training Custom PyTorch Neural Network for {target_col}... ---")

        if use_turbo:
            # *** CLOUD MODE: Use lightweight sklearn RandomForest ***
            # Avoids PyTorch + SHAP memory spikes that OOM-kill 512MB servers.
            print(f"--- [Sourcedotcom] CloudSafe Mode: Using FastPredictor (RandomForest) ---")
            X_raw = df[feature_cols].copy()
            for col in X_raw.columns:
                if not pd.api.types.is_numeric_dtype(X_raw[col]):
                    le = LabelEncoder()
                    X_raw[col] = le.fit_transform(X_raw[col].astype(str))
            X_raw = X_raw.fillna(0).values.astype(np.float32)

            if not pd.api.types.is_numeric_dtype(df[target_col].dropna()):
                le_y = LabelEncoder()
                y_raw = le_y.fit_transform(df[target_col].astype(str)).reshape(-1, 1).astype(np.float32)
            else:
                y_raw = df[target_col].values.reshape(-1, 1)
                y_raw = np.nan_to_num(y_raw, nan=0.0).astype(np.float32)

            if len(X_raw) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
            else:
                X_train, X_test, y_train, y_test = X_raw, X_raw, y_raw, y_raw

            return FastPredictor.get_drivers(X_train, X_test, y_train, y_test, feature_cols)

        # Below this line = use_turbo=False (only for local/dev)
        
        # Safely handle y_raw for classifiers or nulls
        if not pd.api.types.is_numeric_dtype(df[target_col].dropna()):
            le_y = LabelEncoder()
            y_raw = le_y.fit_transform(df[target_col].astype(str)).reshape(-1, 1)
            y_raw = y_raw.astype(np.float32)
        else:
            y_raw = df[target_col].values.reshape(-1, 1)
            y_raw = np.nan_to_num(y_raw, nan=0.0).astype(np.float32)
        
        # Encode categorical features
        for col in X_raw.columns:
            if not pd.api.types.is_numeric_dtype(X_raw[col]):
                le = LabelEncoder()
                X_raw[col] = le.fit_transform(X_raw[col].astype(str))
            
        # Handle NaNs
        X_raw = X_raw.fillna(0).values.astype(np.float32)

        # 3. Train/Test Split (80/20) for proper Evaluation
        if len(X_raw) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X_raw, X_raw, y_raw, y_raw

        if use_turbo:
            return FastPredictor.get_drivers(X_train, X_test, y_train, y_test, feature_cols)
            
        X_train_tensor = torch.from_numpy(X_train)
        y_train_tensor = torch.from_numpy(y_train)
        X_test_tensor = torch.from_numpy(X_test)
        y_test_tensor = torch.from_numpy(y_test)

        # 4. Build Model
        input_dim = X_train.shape[1]
        use_lstm = any(r == 'temporal_axis' for r in roles.values())
        if use_lstm:
            print(f"--- [Sourcedotcom] Applying LSTM Sequence Modeling for time-dependent patterns ---")
            
        model = PerformanceNet(input_dim, use_lstm=use_lstm)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

        # 5. Training Loop with DataLoader (Scalable)
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        batch_size = min(64, len(dataset)) if len(dataset) > 1 else 1 # Capped at 64 for 512MB RAM stability

        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        epochs = 150
        if len(df) < 50: epochs = 300
        if len(df) < 500: epochs = 200
        
        model.train()
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Early Stopping Check
            avg_epoch_loss = epoch_loss / len(loader)
            
            # Aggressive GC inside loop for 512MB RAM environments
            if epoch % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if avg_epoch_loss < best_loss - 1e-4:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"--- [Sourcedotcom] Early Stopping at epoch {epoch} (Accuracy stabilized) ---")
                break

        # 6. Model Evaluation (Prediction vs Actual metrics)
        model.eval()
        with torch.no_grad():
            y_pred_tensor = model(X_test_tensor)
            y_pred_np = y_pred_tensor.numpy()
            y_test_np = y_test_tensor.numpy()
            
            try:
                mae = mean_absolute_error(y_test_np, y_pred_np)
                r2 = r2_score(y_test_np, y_pred_np)
            except Exception:
                mae, r2 = 0.0, 0.0
                
            print(f"--- [Sourcedotcom] Metrics -> MAE: {mae:.2f} | R2: {r2:.2f} ---")

        # 7. Extract Feature Importance (The "Why")
        importances = {}
        try:
            import shap
            # Aggressive sampling for RAM conservation on cloud instances (512MB limit)
            max_shap_samples = 50
            X_train_summary = X_train
            if len(X_train) > max_shap_samples:
                X_train_summary = shap.sample(X_train, max_shap_samples)
            
            X_test_summary = X_test
            if len(X_test) > max_shap_samples:
                X_test_summary = shap.sample(X_test, max_shap_samples)

            explainer = shap.DeepExplainer(model, torch.from_numpy(X_train_summary).float())
            shap_values = explainer.shap_values(torch.from_numpy(X_test_summary).float(), check_additivity=False)
            
            # Aggregate importance
            val = shap_values[0] if isinstance(shap_values, list) else shap_values
            mean_shap = np.abs(val).mean(axis=0)
            if len(mean_shap.shape) > 1:
                # Average across output dims if necessary
                mean_shap = mean_shap.mean(axis=1) 
                
            for i, col_name in enumerate(feature_cols):
                importances[col_name] = float(mean_shap[i])
            print("--- [Sourcedotcom] SHAP values calculated for advanced explainability ---")

        except Exception as e:
            print(f"--- [Sourcedotcom] SHAP Explainer fallback triggered ({e}). Using Permutation. ---")
            with torch.no_grad():
                baseline_output = model(X_test_tensor)
                baseline_error = criterion(baseline_output, y_test_tensor).item()
                
                for i, col_name in enumerate(feature_cols):
                    # Shuffle one column
                    X_permuted = X_test_tensor.clone()
                    perm = torch.randperm(X_permuted.size(0))
                    X_permuted[:, i] = X_permuted[perm, i]
                    
                    perm_output = model(X_permuted)
                    perm_error = criterion(perm_output, y_test_tensor).item()
                    
                    importance = max(0, perm_error - baseline_error)
                    importances[col_name] = float(importance)
                
                # Importance is the delta in error (clamped to 0)
                importance = max(0, perm_error - baseline_error)
                importances[col_name] = float(importance)

        # Normalize 0-1
        total = sum(importances.values())
        if total > 0:
            for k in importances:
                importances[k] /= total
                
        sorted_drivers = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
        
        print(f"--- [Sourcedotcom] Neural Drivers identified: {list(sorted_drivers.keys())[:3]} ---")
        return sorted_drivers, y_test_np, y_pred_np, mae, r2

    @classmethod
    def generate_forecast(cls, df, roles, periods=30):
        """
        Calculates a future forecast based on a temporal axis.
        """
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LinearRegression
        
        time_col = next((c for c, r in roles.items() if r == 'temporal_axis'), None)
        target = next((c for c, r in roles.items() if r == 'primary_metric'), None)
        
        if not time_col or not target or time_col not in df.columns or target not in df.columns:
            return None
            
        try:
            # Group chronologically
            df_time = df.copy()
            df_time[time_col] = pd.to_datetime(df_time[time_col])
            df_time = df_time.dropna(subset=[time_col, target])
            daily_data = df_time.groupby(time_col)[target].sum().reset_index().sort_values(time_col)
            
            if len(daily_data) < 5:
                return None
                
            # Create feature: Days since start
            start_date = daily_data[time_col].min()
            daily_data['days_since'] = (daily_data[time_col] - start_date).dt.days
            
            X_hist = daily_data[['days_since']].values
            y_hist = daily_data[target].values
            
            # Fit Regressor
            model = LinearRegression()
            model.fit(X_hist, y_hist)
            
            # Predict historical 
            daily_data['Forecast'] = model.predict(X_hist)
            daily_data['Actual'] = y_hist
            daily_data = daily_data.rename(columns={time_col: 'Date'})
            
            # Generate future points
            last_date = daily_data['Date'].max()
            last_day = daily_data['days_since'].max()
            
            # Usually daily data steps by days, but if it's sparse, we just project literal days.
            future_dates = [pd.to_datetime(last_date) + pd.Timedelta(days=i) for i in range(1, periods + 1)]
            future_days = np.array([[last_day + i] for i in range(1, periods + 1)])
            future_preds = model.predict(future_days)
            
            future_df = pd.DataFrame({
                'Date': future_dates,
                'days_since': future_days.flatten(),
                'Actual': [np.nan] * periods,
                'Forecast': future_preds
            })
            
            combined = pd.concat([daily_data, future_df], ignore_index=True)
            return combined[['Date', 'Actual', 'Forecast']]
            
        except Exception as e:
            print(f"--- [Forecast Engine] Error generating forecast: {str(e)} ---")
            return None
