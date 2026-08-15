from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gas_daily_prediction.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "gas_ridge.pkl"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
)

FIGURES_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

PREDICTIONS_PATH = (
    RESULTS_PATH
    / "gas_predictions_test.csv"
)



# CARGAR DATASET


def load_dataset():
    """
    Carga el dataset diario utilizado para la predicción.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, sep = ";")


    # Detectar columna temporal


    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

        df = df.sort_values(
            "Date"
        ).reset_index(drop=True)

    elif "Trading_Day" in df.columns:

        df["Trading_Day"] = pd.to_datetime(
            df["Trading_Day"]
        )

        df = df.sort_values(
            "Trading_Day"
        ).reset_index(drop=True)

    print(
        f"Dataset cargado correctamente: "
        f"{df.shape[0]} filas, "
        f"{df.shape[1]} columnas"
    )

    return df



# CARGAR MODELO

def load_model():
    """
    Carga el modelo Ridge previamente entrenado.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo:\n{MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"Modelo cargado correctamente:\n"
        f"{MODEL_PATH}"
    )

    return model



# PREPARAR TEST

def prepare_test_data(
    df,
    target="Gas_Price",
    test_size=0.20
):
    """
    Realiza una separación temporal del dataset.

    El primer 80 % corresponde al entrenamiento
    y el último 20 % al conjunto de test.
    """

    if target not in df.columns:
        raise ValueError(
            f"No existe la variable objetivo '{target}'."
        )

    # Identificar columna temporal

    if "Date" in df.columns:
        date_column = "Date"

    elif "Trading_Day" in df.columns:
        date_column = "Trading_Day"

    else:
        date_column = None

    # Separar variables
  

    excluded_columns = [
        target,
        "Gas_Price_Pct_Change"
    ]

    if date_column is not None:
        excluded_columns.append(
            date_column
        )

    X = df.drop(
        columns=excluded_columns
    )

    y = df[target]


    # Split temporal
   

    split_index = int(
        len(df) * (1 - test_size)
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Fechas del test
    if date_column is not None:

        test_dates = (
            df.iloc[split_index:][date_column]
            .reset_index(drop=True)
        )

    else:

        test_dates = pd.Series(
            range(len(y_test))
        )

    print("\nSeparación temporal:")
    print(
        f"Train: {len(X_train)} observaciones"
    )
    print(
        f"Test:  {len(X_test)} observaciones"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        test_dates
    )


# GENERAR PREDICCIONES

def generate_predictions(
    model,
    X_test
):
    """
    Genera las predicciones sobre el conjunto de test.
    """

    y_pred = model.predict(
        X_test
    )

    return y_pred


# EVALUAR MODELO

def evaluate_predictions(
    y_test,
    y_pred
):
    """
    Calcula MSE, RMSE, MAE y R².
    """

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

    return metrics

def save_metrics(metrics):
    """
    Guarda las métricas de evaluación del modelo.
    """

    metrics_path = (
        RESULTS_PATH
        / "gas_metrics.csv"
    )

    RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        metrics_path,
        sep = ";",
        index=False
    )

    print(
        f"\nMétricas guardadas en:\n"
        f"{metrics_path}"
    )


# GUARDAR PREDICCIONES


def save_predictions(
    test_dates,
    y_test,
    y_pred
):
    """
    Guarda los valores reales, predichos y residuos.
    """

    predictions = pd.DataFrame({
        "Date": test_dates,
        "Actual_Gas_Price": y_test.reset_index(
            drop=True
        ),
        "Predicted_Gas_Price": y_pred
    })

    predictions["Residual"] = (
        predictions["Actual_Gas_Price"]
        - predictions["Predicted_Gas_Price"]
    )

    RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        sep = ";",
        index=False
    )

    print(
        f"\nPredicciones guardadas en:\n"
        f"{PREDICTIONS_PATH}"
    )

    return predictions

# GRÁFICA 1
# REAL VS PREDICHO EN EL TIEMPO


def plot_actual_vs_predicted(
    predictions
):
    """
    Compara el precio real y el precio predicho
    a lo largo del periodo de test.
    """

    FIGURES_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        predictions["Date"],
        predictions["Actual_Gas_Price"],
        label="Precio real"
    )

    plt.plot(
        predictions["Date"],
        predictions["Predicted_Gas_Price"],
        label="Precio predicho"
    )

    plt.xlabel(
        "Fecha"
    )

    plt.ylabel(
        "Precio del gas"
    )

    plt.title(
        "Precio real vs precio predicho del gas"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        FIGURES_PATH
        / "gas_actual_vs_predicted.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Gráfica guardada en:\n"
        f"{output_path}"
    )


# GRÁFICA 2
# RESIDUOS


def plot_residuals(
    predictions
):
    """
    Representa los residuos del modelo a lo largo del tiempo.
    """

    FIGURES_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        predictions["Date"],
        predictions["Residual"]
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Fecha"
    )

    plt.ylabel(
        "Residuo"
    )

    plt.title(
        "Residuos del modelo de predicción del precio del gas"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        FIGURES_PATH
        / "gas_prediction_residuals.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Gráfica guardada en:\n"
        f"{output_path}"
    )


# GRÁFICA 3
# REAL VS PREDICHO — SCATTER


def plot_real_vs_predicted_scatter(
    predictions
):
    """
    Scatter de precios reales frente a precios predichos.
    """

    FIGURES_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    actual = predictions[
        "Actual_Gas_Price"
    ]

    predicted = predictions[
        "Predicted_Gas_Price"
    ]

    plt.figure(
        figsize=(8, 8)
    )

    plt.scatter(
        actual,
        predicted,
        alpha=0.6
    )

    # Línea ideal y = x
  

    min_value = min(
        actual.min(),
        predicted.min()
    )

    max_value = max(
        actual.max(),
        predicted.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Predicción ideal"
    )

    plt.xlabel(
        "Precio real"
    )

    plt.ylabel(
        "Precio predicho"
    )

    plt.title(
        "Precio real vs precio predicho"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        FIGURES_PATH
        / "gas_real_vs_predicted_scatter.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Gráfica guardada en:\n"
        f"{output_path}"
    )



# MAIN


def main():

    print("=" * 60)
    print("PREDICCIÓN DEL PRECIO DIARIO DEL GAS")
    print("=" * 60)

   
    # 1. Cargar dataset

    df = load_dataset()

    # 2. Cargar modelo

    model = load_model()

    # 3. Preparar test
   
    (
        _,
        X_test,
        _,
        y_test,
        test_dates
    ) = prepare_test_data(
        df
    )

    # 4. Generar predicciones


    print(
        "\nGenerando predicciones..."
    )

    y_pred = generate_predictions(
        model,
        X_test
    )

    # 5. Evaluar


    metrics = evaluate_predictions(
        y_test,
        y_pred
    )

    save_metrics(metrics)

    print(
        "\nRESULTADOS DEL MODELO"
    )

    print("-" * 60)

    for metric, value in metrics.items():

        print(
            f"{metric:<10}: {value:.6f}"
        )

    # 6. Guardar predicciones


    predictions = save_predictions(
        test_dates,
        y_test,
        y_pred
    )

    # 7. Gráfica real vs predicho


    plot_actual_vs_predicted(
        predictions
    )

  
    # 8. Gráfica de residuos
   

    plot_residuals(
        predictions
    )

    # 9. Scatter real vs predicho
   

    plot_real_vs_predicted_scatter(
        predictions
    )

    print(
        "\nProceso de predicción completado."
    )


if __name__ == "__main__":
    main()