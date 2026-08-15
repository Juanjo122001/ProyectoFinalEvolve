from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# ============================================================
# RUTAS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gas_daily_prediction.csv"
)

MODELS_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

MODELS_PATH.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CARGAR DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Carga el dataset diario de MIBGAS.
    """

    df = pd.read_csv(
        DATA_PATH,
        sep=";"
    )

    df["Trading_Day"] = pd.to_datetime(
        df["Trading_Day"]
    )

    df = df.sort_values(
        "Trading_Day"
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# PREPARAR X / y
# ============================================================

def prepare_data(
    df: pd.DataFrame
):
    """
    Separa variables predictoras y variable objetivo.

    Target:
        Gas_Price

    No se utiliza Trading_Day directamente como predictor.
    """

    target = "Gas_Price"

    # Variables que no deben entrar directamente en X
    excluded_columns = [
        "Trading_Day",
        target,
        "Gas_Price_Pct_Change",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    X = df[feature_columns].copy()
    y = df[target].copy()

    return X, y, feature_columns


# ============================================================
# SPLIT TEMPORAL
# ============================================================

def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20
):
    """
    Divide los datos respetando el orden temporal.

    El 80 % inicial se utiliza para entrenamiento y
    el 20 % final para evaluación.
    """

    split_index = int(
        len(X) * (1 - test_size)
    )

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# MODELOS
# ============================================================

def create_models():
    """
    Crea los modelos de regresión que se compararán.
    """

    models = {

        "Linear Regression":
            Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    LinearRegression()
                )
            ]),

        "Ridge Regression":
            Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    Ridge(alpha=1.0)
                )
            ]),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            ),

        "XGBoost":
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            ),

        "LightGBM":
            LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ),

        "CatBoost":
            CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                loss_function="RMSE",
                random_seed=42,
                verbose=False
            )
    }

    return models


# ============================================================
# EVALUACIÓN
# ============================================================

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Entrena un modelo y calcula las métricas de evaluación.
    """

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mse
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return (
        model,
        predictions,
        mse,
        rmse,
        mae,
        r2
    )


# ============================================================
# ENTRENAMIENTO
# ============================================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_columns
):
    """
    Entrena todos los modelos, calcula sus métricas y
    guarda los modelos entrenados.
    """

    models = create_models()

    results = []

    for model_name, model in models.items():

        print("\n" + "-" * 60)
        print(
            f"Entrenando: {model_name}"
        )
        print("-" * 60)

        (
            trained_model,
            predictions,
            mse,
            rmse,
            mae,
            r2
        ) = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print(
            f"MSE:  {mse:.6f}"
        )

        print(
            f"RMSE: {rmse:.6f}"
        )

        print(
            f"MAE:  {mae:.6f}"
        )

        print(
            f"R²:   {r2:.6f}"
        )

        # ----------------------------------------------------
        # Nombre de archivo
        # ----------------------------------------------------

        filename = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("regression", "")
            .strip("_")
        )

        model_path = (
            MODELS_PATH
            / f"gas_{filename}.pkl"
        )

        joblib.dump(
            trained_model,
            model_path
        )

        print(
            f"Modelo guardado en: "
            f"{model_path}"
        )

        results.append({
            "Model": model_name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "RMSE"
    ).reset_index(
        drop=True
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PREDICCIÓN DIARIA DEL PRECIO DEL GAS MIBGAS")
    print("=" * 60)

    # --------------------------------------------------------
    # Cargar datos
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"\nRegistros: {len(df)}"
    )

    print(
        f"Periodo: "
        f"{df['Trading_Day'].min().date()} "
        f"→ "
        f"{df['Trading_Day'].max().date()}"
    )

    # --------------------------------------------------------
    # Preparar X / y
    # --------------------------------------------------------

    X, y, feature_columns = prepare_data(
        df
    )

    print(
        f"\nNúmero de variables predictoras: "
        f"{len(feature_columns)}"
    )

    print("\nVariables predictoras:")

    for feature in feature_columns:
        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # Split temporal
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = temporal_train_test_split(
        X,
        y
    )

    print("\n" + "=" * 60)
    print("SPLIT TEMPORAL")
    print("=" * 60)

    print(
        f"\nTrain: {len(X_train)} registros"
    )

    print(
        f"Test:  {len(X_test)} registros"
    )

    print(
        f"\nTrain:"
        f" {df['Trading_Day'].iloc[0].date()}"
        f" → "
        f"{df['Trading_Day'].iloc[len(X_train)-1].date()}"
    )

    print(
        f"Test:"
        f" {df['Trading_Day'].iloc[len(X_train)].date()}"
        f" → "
        f"{df['Trading_Day'].iloc[-1].date()}"
    )

    # --------------------------------------------------------
    # Entrenar modelos
    # --------------------------------------------------------

    results_df = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns
    )

    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Guardar resultados
    # --------------------------------------------------------

    results_path = (
        METRICS_PATH
        / "gas_model_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(
        f"\nResultados guardados en:"
        f"\n{results_path}"
    )


if __name__ == "__main__":
    main()