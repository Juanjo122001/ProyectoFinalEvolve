from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


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

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
)

FIGURES_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

FIGURES_PATH.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_ANALYZE = {
    "Ridge" : "gas_ridge.pkl",
    "LightGBM" : "gas_lightgbm.pkl"
}


# ============================================================
# CARGAR DATOS
# ============================================================

def load_data():

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
# PREPARAR X
# ============================================================

def prepare_features(df):

    target = "Gas_Price"

    excluded_columns = [
        "Trading_Day",
        target,
        "Gas_Price_Pct_Change"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    X = df[feature_columns].copy()

    return X


# ============================================================
# CARGAR MODELO
# ============================================================

def load_model(model_filename):

    model_path = MODEL_PATH / model_filename

    model = joblib.load(
        model_path
    )

    return model


# ============================================================
# CALCULAR SHAP
# ============================================================

def calculate_shap_values(
    model,
    X,
    model_name
):

    # --------------------------------------------------------
    # En Ridge tenemos un Pipeline:
    #
    # StandardScaler
    # +
    # Ridge
    #
    # SHAP debe analizar el modelo completo.
    # --------------------------------------------------------

    if "LightGBM" in model_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
    else:
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)

    return shap_values


# ============================================================
# SHAP SUMMARY
# ============================================================

def plot_shap_summary(
    shap_values,
    X,
    model_name
):

    plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    plt.tight_layout()

    filename_suffix = model_name.lower().replace(" ", "_")

    output_path = (
        FIGURES_PATH
        / f"shap_summary_gas_{filename_suffix}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Gráfico SHAP summary guardado en:"
        f"\n{output_path}"
    )


# ============================================================
# SHAP BAR
# ============================================================

def plot_shap_bar(
    shap_values,
    X,
    model_name
):

    plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    filename_suffix = model_name.lower().replace(" ", "_")

    output_path = (
        FIGURES_PATH
        / f"shap_importance_gas_{filename_suffix}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Gráfico SHAP bar guardado en:"
        f"\n{output_path}"
    )


# ============================================================
# DEPENDENCIA SHAP
# ============================================================

def plot_shap_dependence(
    shap_values,
    X,
    feature,
    model_name
):
    values_to_plot = shap_values.values if hasattr(shap_values, 'values') else shap_values
    

    plt.figure()

    shap.dependence_plot(
        feature,
        values_to_plot,
        X,
        show=False
    )

    plt.tight_layout()

    filename_suffix = model_name.lower().replace(" ", "_")


    output_path = (
        FIGURES_PATH
        / f"shap_dependence_{feature.lower()}_{filename_suffix}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Gráfico de dependencia guardado en:"
        f"\n{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ANÁLISIS SHAP — PRECIO DIARIO DEL GAS")
    print("=" * 60)

    # --------------------------------------------------------
    # Datos
    # --------------------------------------------------------

    df = load_data()

    print(
        f"\nRegistros: {len(df)}"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = prepare_features(
        df
    )

    print(
        f"Variables analizadas: {len(X.columns)}"
    )

    # --------------------------------------------------------
    # Modelo

    for model_name, model_filename in MODELS_ANALYZE.items():
        print(f"Analizando modelo: {model_name}")

        model = load_model(model_filename)
        if model is None:
            continue

        print(f"Calculando valores SHAP para {model_name}")
        shap_values = calculate_shap_values(model, X, model_name)
        print("Valores SHAP calculados correctamente")
        plot_shap_summary(shap_values,X,model_name)
        plot_shap_bar(shap_values,X,model_name)
        if "Gas_Price_Lag_1" in X.columns:
            print("\nGenerando gráfico de dependencia para 'Gas_Price_Lag_1'...")
            plot_shap_dependence(shap_values,X,"Gas_Price_Lag_1", model_name)

    print("\nAnálisis SHAP completado.")


if __name__ == "__main__":
    main()