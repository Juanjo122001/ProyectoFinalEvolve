from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

FORECAST_DAYS = 30
TARGET = "Gas_Price"


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

MODELS_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
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

FUTURE_PREDICTIONS_PATH = (
    RESULTS_PATH
    / "gas_future_predictions_comparison.csv"
)


# Modelos utilizados para la predicción futura.
# Si el archivo LightGBM tiene otro nombre, modificar aquí.
MODELS = {
    "Ridge Regression": "gas_ridge.pkl",
    "LightGBM": "gas_lightgbm.pkl",
}


# ============================================================
# CARGAR DATASET
# ============================================================

def load_dataset():
    """
    Carga el dataset diario utilizado para la predicción.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        sep=";"
    )

    if "Trading_Day" not in df.columns:
        raise ValueError(
            "El dataset no contiene la columna 'Trading_Day'."
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"El dataset no contiene la variable objetivo "
            f"'{TARGET}'."
        )

    df["Trading_Day"] = pd.to_datetime(
        df["Trading_Day"]
    )

    df = (
        df
        .sort_values("Trading_Day")
        .reset_index(drop=True)
    )

    print(
        f"Dataset cargado correctamente: "
        f"{df.shape[0]} filas, "
        f"{df.shape[1]} columnas"
    )

    print(
        f"Periodo: "
        f"{df['Trading_Day'].min().date()} → "
        f"{df['Trading_Day'].max().date()}"
    )

    return df


# ============================================================
# CARGAR MODELOS
# ============================================================

def load_models():
    """
    Carga todos los modelos utilizados para la predicción futura.

    Para cada modelo se recupera también el orden exacto de las
    variables utilizado durante el entrenamiento. Esto evita el
    error de scikit-learn relacionado con el orden de features.
    """

    models = {}

    for model_name, model_filename in MODELS.items():

        model_path = (
            MODELS_PATH
            / model_filename
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo "
                f"{model_name}:\n{model_path}"
            )

        model = joblib.load(
            model_path
        )

        print(
            f"\nModelo {model_name} cargado correctamente:"
        )
        print(model_path)

        # --------------------------------------------------------
        # Recuperar el orden de features
        # --------------------------------------------------------

        if hasattr(
            model,
            "feature_names_in_"
        ):
            expected_features = (
                model.feature_names_in_.tolist()
            )

        elif hasattr(
            model,
            "feature_name_"
        ):
            expected_features = list(
                model.feature_name_
            )

        elif hasattr(
            model,
            "named_steps"
        ):

            expected_features = None

            # Buscar un paso del pipeline que conserve
            # feature_names_in_.
            for _, step in model.named_steps.items():

                if hasattr(
                    step,
                    "feature_names_in_"
                ):
                    expected_features = (
                        step.feature_names_in_.tolist()
                    )
                    break

            if expected_features is None:
                raise ValueError(
                    f"No se ha podido recuperar el orden "
                    f"de las features del modelo "
                    f"{model_name}."
                )

        else:
            raise ValueError(
                f"El modelo {model_name} no contiene información "
                f"sobre las features utilizadas durante el fit."
            )

        print(
            f"\nOrden de features utilizado por "
            f"{model_name}:"
        )

        for i, feature in enumerate(
            expected_features,
            start=1
        ):
            print(
                f"{i:02d}. {feature}"
            )

        models[model_name] = {
            "model": model,
            "expected_features": expected_features,
        }

    return models


# ============================================================
# CREAR VARIABLES TEMPORALES
# ============================================================

def create_temporal_features(df):
    """
    Crea las variables temporales utilizadas por el modelo.
    """

    df = df.copy()

    df["Day"] = (
        df["Trading_Day"].dt.day
    )

    df["Month"] = (
        df["Trading_Day"].dt.month
    )

    df["Year"] = (
        df["Trading_Day"].dt.year
    )

    df["Quarter"] = (
        df["Trading_Day"].dt.quarter
    )

    df["Day_of_Year"] = (
        df["Trading_Day"].dt.dayofyear
    )

    df["Day_of_Week"] = (
        df["Trading_Day"].dt.dayofweek
    )

    df["Semester_Number"] = (
        (df["Month"] - 1) // 6 + 1
    )

    # Variables cíclicas

    df["Month_Sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )

    df["Month_Cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )

    df["Day_of_Week_Sin"] = np.sin(
        2 * np.pi * df["Day_of_Week"] / 7
    )

    df["Day_of_Week_Cos"] = np.cos(
        2 * np.pi * df["Day_of_Week"] / 7
    )

    return df


# ============================================================
# CREAR VARIABLES LAG
# ============================================================

def create_lag_features(df):
    """
    Crea las variables lag del precio del gas.
    """

    df = df.copy()

    df["Gas_Price_Lag_1"] = (
        df["Gas_Price"].shift(1)
    )

    df["Gas_Price_Lag_7"] = (
        df["Gas_Price"].shift(7)
    )

    df["Gas_Price_Lag_14"] = (
        df["Gas_Price"].shift(14)
    )

    df["Gas_Price_Lag_30"] = (
        df["Gas_Price"].shift(30)
    )

    return df


# ============================================================
# VARIABLES ROLLING
# ============================================================

def create_rolling_features(df):
    """
    Crea estadísticas móviles del precio del gas.
    """

    df = df.copy()

    df["Gas_Rolling_Mean_7"] = (
        df["Gas_Price"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    df["Gas_Rolling_Mean_30"] = (
        df["Gas_Price"]
        .shift(1)
        .rolling(window=30)
        .mean()
    )

    df["Gas_Rolling_Std_7"] = (
        df["Gas_Price"]
        .shift(1)
        .rolling(window=7)
        .std()
    )

    return df


# ============================================================
# TENDENCIA
# ============================================================

def create_trend_feature(df):
    """
    Calcula la tendencia reciente del precio del gas.
    """

    df = df.copy()

    df["Gas_Trend"] = (
        df["Gas_Price_Lag_1"]
        - df["Gas_Price_Lag_7"]
    )

    return df


# ============================================================
# CAMBIO PORCENTUAL
# ============================================================

def create_pct_change_feature(df):
    """
    Calcula el cambio porcentual del precio respecto
    al día anterior utilizando información disponible.
    """

    df = df.copy()

    df["Gas_Price_Pct_Change_Lag_1"] = (
        df["Gas_Price_Lag_1"]
        .pct_change()
    )

    return df


# ============================================================
# VARIABLES GEOPOLÍTICAS
# ============================================================

def add_geopolitical_features(df):
    """
    Añade variables relacionadas con eventos geopolíticos
    en función de la fecha.
    """

    df = df.copy()

    # --------------------------------------------------------
    # GUERRA DE UCRANIA
    # --------------------------------------------------------

    df["Ukraine_War"] = (
        df["Trading_Day"]
        >= pd.Timestamp("2022-02-24")
    ).astype(int)

    # --------------------------------------------------------
    # CRISIS ENERGÉTICA
    # --------------------------------------------------------

    df["Energy_Crisis"] = (
        (
            df["Trading_Day"]
            >= pd.Timestamp("2021-09-01")
        )
        &
        (
            df["Trading_Day"]
            <= pd.Timestamp("2023-03-31")
        )
    ).astype(int)

    # --------------------------------------------------------
    # CONFLICTO DE ORIENTE MEDIO
    # --------------------------------------------------------

    df["Middle_East_Conflict"] = (
        df["Trading_Day"]
        >= pd.Timestamp("2023-10-07")
    ).astype(int)

    return df


# ============================================================
# CREAR TODAS LAS VARIABLES PREDICTIVAS
# ============================================================

def create_prediction_features(df):
    """
    Crea todas las variables necesarias para el modelo
    de predicción futura.
    """

    df = df.copy()

    df = create_temporal_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_trend_feature(df)
    df = create_pct_change_feature(df)
    df = add_geopolitical_features(df)

    return df


# ============================================================
# PREPARAR FILA FUTURA
# ============================================================

def prepare_future_row(
    history,
    future_date
):
    """
    Construye las variables predictivas para una fecha futura.

    history contiene tanto precios reales conocidos como
    precios predichos anteriormente.
    """

    future_date = pd.Timestamp(
        future_date
    )

    prices = (
        history
        .set_index("Trading_Day")["Gas_Price"]
        .sort_index()
    )

    if len(prices) < 30:
        raise ValueError(
            "Se necesitan al menos 30 observaciones "
            "históricas para generar las variables lag."
        )

    # --------------------------------------------------------
    # VARIABLES TEMPORALES
    # --------------------------------------------------------

    row = pd.DataFrame({
        "Trading_Day": [future_date],

        "Day": [future_date.day],

        "Month": [future_date.month],

        "Year": [future_date.year],

        "Quarter": [
            future_date.quarter
        ],

        "Day_of_Year": [
            future_date.dayofyear
        ],

        "Day_of_Week": [
            future_date.dayofweek
        ],

        "Semester_Number": [
            (future_date.month - 1) // 6 + 1
        ],

        "Month_Sin": [
            np.sin(
                2 * np.pi
                * future_date.month
                / 12
            )
        ],

        "Month_Cos": [
            np.cos(
                2 * np.pi
                * future_date.month
                / 12
            )
        ],

        "Day_of_Week_Sin": [
            np.sin(
                2 * np.pi
                * future_date.dayofweek
                / 7
            )
        ],

        "Day_of_Week_Cos": [
            np.cos(
                2 * np.pi
                * future_date.dayofweek
                / 7
            )
        ],
    })

    # --------------------------------------------------------
    # VARIABLES LAG
    # --------------------------------------------------------

    row["Gas_Price_Lag_1"] = (
        prices.iloc[-1]
    )

    row["Gas_Price_Lag_7"] = (
        prices.iloc[-7]
    )

    row["Gas_Price_Lag_14"] = (
        prices.iloc[-14]
    )

    row["Gas_Price_Lag_30"] = (
        prices.iloc[-30]
    )

    # --------------------------------------------------------
    # ROLLING
    # --------------------------------------------------------

    row["Gas_Rolling_Mean_7"] = (
        prices.iloc[-7:].mean()
    )

    row["Gas_Rolling_Mean_30"] = (
        prices.iloc[-30:].mean()
    )

    row["Gas_Rolling_Std_7"] = (
        prices.iloc[-7:].std()
    )

    # --------------------------------------------------------
    # TENDENCIA
    # --------------------------------------------------------

    row["Gas_Trend"] = (
        row["Gas_Price_Lag_1"].iloc[0]
        - row["Gas_Price_Lag_7"].iloc[0]
    )

    # --------------------------------------------------------
    # CAMBIO PORCENTUAL LAG
    # --------------------------------------------------------

    row["Gas_Price_Pct_Change_Lag_1"] = (
        prices.iloc[-1]
        / prices.iloc[-2]
        - 1
    )

    # --------------------------------------------------------
    # VARIABLES GEOPOLÍTICAS
    # --------------------------------------------------------

    row = add_geopolitical_features(
        row
    )

    return row


# ============================================================
# PREDICCIÓN FUTURA RECURSIVA
# ============================================================

def recursive_forecast(
    df,
    model,
    expected_features,
    forecast_days=30,
    model_name="Modelo"
):
    """
    Genera predicciones futuras de forma recursiva.

    Cada predicción generada pasa a formar parte del histórico
    utilizado para calcular los lags de los días siguientes.

    Cada modelo recibe su propio histórico recursivo.
    """

    history = df[
        ["Trading_Day", "Gas_Price"]
    ].copy()

    history["Trading_Day"] = pd.to_datetime(
        history["Trading_Day"]
    )

    history = (
        history
        .sort_values("Trading_Day")
        .reset_index(drop=True)
    )

    predictions = []

    last_date = (
        history["Trading_Day"].iloc[-1]
    )

    print(
        f"\nÚltima fecha conocida: "
        f"{last_date.date()}"
    )

    print(
        f"Generando {forecast_days} días de "
        f"predicción con {model_name}..."
    )

    for step in range(
        1,
        forecast_days + 1
    ):

        future_date = (
            last_date
            + pd.Timedelta(days=step)
        )

        # ----------------------------------------------------
        # Construir fila futura
        # ----------------------------------------------------

        future_row = prepare_future_row(
            history,
            future_date
        )

        # ----------------------------------------------------
        # Variables utilizadas por el modelo
        # ----------------------------------------------------

        X_future = future_row.drop(
            columns=["Trading_Day"],
            errors="ignore"
        )

        missing_features = [
            feature
            for feature in expected_features
            if feature not in X_future.columns
        ]

        extra_features = [
            feature
            for feature in X_future.columns
            if feature not in expected_features
        ]

        if missing_features:
            raise ValueError(
                f"Faltan variables esperadas por "
                f"el modelo {model_name}:\n"
                f"{missing_features}"
            )

        if extra_features:
            raise ValueError(
                f"Hay variables que no fueron utilizadas "
                f"durante el entrenamiento de {model_name}:\n"
                f"{extra_features}"
            )

        # ----------------------------------------------------
        # MUY IMPORTANTE:
        # Mantener exactamente el orden del entrenamiento
        # ----------------------------------------------------

        X_future = X_future[
            expected_features
        ]

        # ----------------------------------------------------
        # Predicción
        # ----------------------------------------------------

        predicted_price = model.predict(
            X_future
        )[0]

        # ----------------------------------------------------
        # Guardar predicción
        # ----------------------------------------------------

        predictions.append({
            "Trading_Day": future_date,
            "Predicted_Gas_Price": predicted_price
        })

        # ----------------------------------------------------
        # Añadir predicción al histórico
        # ----------------------------------------------------

        new_row = pd.DataFrame({
            "Trading_Day": [future_date],
            "Gas_Price": [predicted_price]
        })

        history = pd.concat(
            [
                history,
                new_row
            ],
            ignore_index=True
        )

        print(
            f"{future_date.date()} → "
            f"{predicted_price:.2f}"
        )

    return pd.DataFrame(
        predictions
    )


# ============================================================
# GENERAR PREDICCIONES DE TODOS LOS MODELOS
# ============================================================

def generate_future_predictions(
    df,
    models,
    forecast_days=30
):
    """
    Genera predicciones futuras independientes para todos
    los modelos.

    Cada modelo utiliza:
    - su propio objeto entrenado;
    - su propio expected_features;
    - su propio histórico recursivo.

    Devuelve un DataFrame con una columna de predicción
    independiente para cada modelo.
    """

    all_predictions = None

    for model_name, model_info in models.items():

        model = model_info["model"]

        expected_features = (
            model_info["expected_features"]
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"PREDICCIÓN FUTURA - {model_name}"
        )

        print(
            "=" * 60
        )

        predictions = recursive_forecast(
            df=df,
            model=model,
            expected_features=expected_features,
            forecast_days=forecast_days,
            model_name=model_name
        )

        prediction_column = (
            model_name
            .replace(" ", "_")
            + "_Prediction"
        )

        predictions = predictions.rename(
            columns={
                "Predicted_Gas_Price":
                prediction_column
            }
        )

        if all_predictions is None:

            all_predictions = predictions

        else:

            all_predictions = all_predictions.merge(
                predictions,
                on="Trading_Day",
                how="inner"
            )

    return all_predictions


# ============================================================
# GUARDAR PREDICCIONES FUTURAS
# ============================================================

def save_future_predictions(
    predictions
):
    """
    Guarda en un único CSV las predicciones futuras
    de todos los modelos.
    """

    RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions.to_csv(
        FUTURE_PREDICTIONS_PATH,
        index=False
    )

    print(
        "\nPredicciones futuras guardadas en:"
    )

    print(
        FUTURE_PREDICTIONS_PATH
    )


# ============================================================
# GRÁFICA HISTÓRICO + PREDICCIÓN FUTURA
# ============================================================

def plot_future_forecast(
    df,
    predictions,
    model_name,
    prediction_column
):
    """
    Genera una gráfica independiente para cada modelo.
    """

    FIGURES_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(14, 7)
    )

    # --------------------------------------------------------
    # Histórico
    # --------------------------------------------------------

    plt.plot(
        df["Trading_Day"],
        df["Gas_Price"],
        label="Precio histórico"
    )

    # --------------------------------------------------------
    # Predicción futura
    # --------------------------------------------------------

    plt.plot(
        predictions["Trading_Day"],
        predictions[prediction_column],
        linestyle="--",
        label=f"Predicción futura - {model_name}"
    )

    # --------------------------------------------------------
    # Línea de separación
    # --------------------------------------------------------

    last_date = (
        df["Trading_Day"].max()
    )

    plt.axvline(
        last_date,
        linestyle=":",
        label="Inicio de predicción"
    )

    # --------------------------------------------------------
    # Etiquetas
    # --------------------------------------------------------

    plt.xlabel(
        "Fecha"
    )

    plt.ylabel(
        "Precio del gas (€/MWh)"
    )

    plt.title(
        "Predicción futura del precio diario del gas - "
        f"{model_name}"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + "_gas_future_forecast.png"
    )

    output_path = (
        FIGURES_PATH
        / filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nGráfica de {model_name} guardada en:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "PREDICCIÓN FUTURA DEL PRECIO DEL GAS"
    )

    print(
        "-" * 60
    )

    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # 2. Cargar modelos
    # --------------------------------------------------------

    models = load_models()

    # --------------------------------------------------------
    # 3. Generar predicciones futuras
    # --------------------------------------------------------

    predictions = generate_future_predictions(
        df=df,
        models=models,
        forecast_days=FORECAST_DAYS
    )

    # --------------------------------------------------------
    # 4. Guardar predicciones
    # --------------------------------------------------------

    save_future_predictions(
        predictions
    )

    # --------------------------------------------------------
    # 5. Generar gráficas independientes
    # --------------------------------------------------------

    plot_future_forecast(
        df=df,
        predictions=predictions,
        model_name="Ridge Regression",
        prediction_column="Ridge_Regression_Prediction"
    )

    plot_future_forecast(
        df=df,
        predictions=predictions,
        model_name="LightGBM",
        prediction_column="LightGBM_Prediction"
    )

    print(
        "\nPredicción futura completada correctamente "
        "para ambos modelos."
    )


if __name__ == "__main__":
    main()