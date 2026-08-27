import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PATH CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "gas_daily_prediction.csv"
MODELS_PATH = PROJECT_ROOT / "results" / "models"
RESULTS_PATH = PROJECT_ROOT / "results" / "tables"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"

# Create directories if they do not exist
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

# Dictionary with the models to evaluate
MODELS = {
    "Ridge Regression": "gas_ridge.pkl",
    "LightGBM": "gas_lightgbm.pkl"
}

def load_data() -> pd.DataFrame:
    """Load the daily gas dataset and ensure dates are parsed correctly."""
    df = pd.read_csv(DATA_PATH, sep=";")
    df["Trading_Day"] = pd.to_datetime(df["Trading_Day"])
    return df.sort_values("Trading_Day").reset_index(drop=True)

def get_expected_features(model) -> list:
    """
    Extract the exact feature order used during the model training phase.
    Handles both Scikit-Learn pipelines and direct LightGBM/XGBoost models.
    """
    # 1. Standard Scikit-Learn estimators
    if hasattr(model, "feature_names_in_"):
        return model.feature_names_in_.tolist()
        
    # 2. LightGBM specific attribute
    elif hasattr(model, "feature_name_"):
        return list(model.feature_name_)
        
    # 3. Scikit-Learn Pipelines (like Ridge Regression)
    elif hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return step.feature_names_in_.tolist()
            elif hasattr(step, "feature_name_"):
                return list(step.feature_name_)
                
    # If we reach here, raise an error instead of returning [] silently
    raise ValueError(f"Could not extract feature names from {type(model).__name__}")

def run_recursive_backtest(df: pd.DataFrame, start_idx: int, model, expected_features: list) -> list:
    """
    Run a 30-day recursive forecast starting from start_idx.
    Uses its own previous predictions to calculate future lags and rolling means.
    """
    # Historical actual prices up to the start index
    history = df.loc[:start_idx - 1, "Gas_Price"].tolist()
    predictions = []

    for step in range(30):
        current_idx = start_idx + step
        
        # Stop if we hit the end of the dataset bounds
        if current_idx >= len(df):
            break
            
        # Get static features (dates, geopolitical dummies) from the actual dataset
        X_step = df.iloc[[current_idx]].copy()
        
        # Overwrite dynamic features using our own predicted history to avoid Data Leakage
        X_step['Gas_Price_Lag_1'] = history[-1]
        X_step['Gas_Price_Lag_7'] = history[-7]
        X_step['Gas_Price_Lag_14'] = history[-14]
        X_step['Gas_Price_Lag_30'] = history[-30]
        X_step['Gas_Rolling_Mean_7'] = np.mean(history[-7:])
        X_step['Gas_Rolling_Mean_30'] = np.mean(history[-30:])
        X_step['Gas_Rolling_Std_7'] = np.std(history[-7:], ddof=1)
        X_step['Gas_Trend'] = history[-1] - history[-7]
        
        # Handle division by zero carefully for percentage change
        X_step['Gas_Price_Pct_Change_Lag_1'] = (history[-1] / history[-2]) - 1 if history[-2] != 0 else 0
        
        # Ensure exact feature order for the model
        X_features = X_step[expected_features]
        
        # Predict next day and append to our simulated history
        pred = model.predict(X_features)[0]
        predictions.append(pred)
        history.append(pred)
        
    return predictions

def main():
    print("=" * 70)
    print("30-DAY RECURSIVE BACKTESTING (ALL METRICS)")
    print("=" * 70)

    df = load_data()
    
    # Identify test set (last 20% of the dataset)
    test_size = int(len(df) * 0.20)
    test_start_idx = len(df) - test_size
    
    # Select multiple evenly spaced origins in the test set to run 30-day simulations
    # Ensuring there are at least 30 days left after the origin
    origins = np.linspace(test_start_idx, len(df) - 31, 5, dtype=int)
    
    horizons = [1, 7, 14, 30]
    results = []

    # Load machine learning models
    loaded_models = {}
    for name, filename in MODELS.items():
        model_path = MODELS_PATH / filename
        if not model_path.exists():
            print(f"Warning: {filename} not found. Please train it first.")
            continue
        model = joblib.load(model_path)
        loaded_models[name] = {
            "model": model,
            "features": get_expected_features(model)
        }
        
    print(f"Running backtest from {len(origins)} different origins in the test set...\n")

    for origin_idx in origins:
        origin_date = df.loc[origin_idx, "Trading_Day"].date()
        
        # Get actual target values for the next 30 days
        actuals = df.loc[origin_idx:origin_idx+29, "Gas_Price"].values
        
        # 1. NAIVE BASELINE (Price tomorrow is the same as price today)
        last_known_price = df.loc[origin_idx - 1, "Gas_Price"]
        naive_preds = np.full(30, last_known_price)
        
        # Save Naive predictions
        for h in horizons:
            results.append({
                "Origin": origin_date, "Model": "Naive", "Horizon": h, 
                "Actual": actuals[h-1], "Predicted": naive_preds[h-1]
            })
            
        # 2. MACHINE LEARNING MODELS
        for model_name, info in loaded_models.items():
            preds = run_recursive_backtest(df, origin_idx, info["model"], info["features"])
            
            # Save ML predictions
            for h in horizons:
                results.append({
                    "Origin": origin_date, "Model": model_name, "Horizon": h, 
                    "Actual": actuals[h-1], "Predicted": preds[h-1]
                })

    # Convert predictions to a DataFrame to compute metrics by Model and Horizon
    results_df = pd.DataFrame(results)
    metrics_list = []
    
    for (model, horizon), group in results_df.groupby(["Model", "Horizon"]):
        y_true = group["Actual"]
        y_pred = group["Predicted"]
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        # Calculate R2 only if there is variance in y_true
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 and np.var(y_true) > 0 else np.nan
        
        metrics_list.append({
            "Model": model,
            "Horizon_Days": horizon,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })
        
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.sort_values(by=["Horizon_Days", "Model"]).reset_index(drop=True)
    
    # Print the table nicely in the terminal
    print("MÉTRICAS DE EVALUACIÓN EN BACKTESTING:")
    print("-" * 70)
    print(metrics_df.round(4).to_string(index=False))
    
    # Save metrics table to CSV
    csv_path = RESULTS_PATH / "backtesting_all_metrics.csv"
    metrics_df.to_csv(csv_path, sep=";", index=False)
    print(f"\nMetrics saved to: {csv_path}")

    # Generate Error Degradation Plot (using RMSE as it penalizes large errors)
    plt.figure(figsize=(10, 6))
    
    # Pivot the data specifically for the plot
    plot_data = metrics_df.pivot(index="Model", columns="Horizon_Days", values="RMSE")
    
    for model in plot_data.index:
        plt.plot(horizons, plot_data.loc[model], marker='o', linewidth=2, label=model)
        
    plt.title("RMSE Degradación con el tiempo (30-Day Recursive Forecast)", fontsize=14)
    plt.xlabel("Horizonte de predicción (Días)", fontsize=12)
    plt.ylabel("Root Mean Squared Error (RMSE) [€/MWh]", fontsize=12)
    plt.xticks(horizons)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = FIGURES_PATH / "backtesting_rmse_degradation.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()