import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from preprocessing import prepare_ml_dataset
from dataset_split import temporal_train_test_split

def load_model(model_path: Path):
    """
    Load a pre-trained model from the specified path.

    Args:
        model_path (Path): Path to the saved model file.

    Returns:
        object: The loaded model.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"No se ha encontrado el modelo en: {model_path}")
    model = joblib.load(model_path)
    return model

def prepare_test_data(
        processed_data_path: Path
):
    """"
    Prepara los datos de prueba para el análisis de interpretabilidad.

    Parameters:
        processed_data_path (Path): Ruta al archivo CSV con los datos procesados.

    Returns:
        X_test (pd.DataFrame): Conjunto de características de prueba.
        y_test (pd.Series): Conjunto de etiquetas de prueba.
    """

    X, y = prepare_ml_dataset(
        processed_data_path
    )

    _, X_test, _, y_test = (
        temporal_train_test_split(
            X,
            y
        )
    )

    return X_test, y_test


def create_shap_explainer(model):
    """
    Crea un objeto SHAP explainer para el modelo dado y los datos de prueba.

    Parameters:
        model: Modelo entrenado.
        X_test (pd.DataFrame): Conjunto de características de prueba.

    Returns:
        shap.Explainer: Objeto SHAP explainer.
    """
    explainer = shap.TreeExplainer(model)
    return explainer


def calculate_shap_values(explainer, X_test):
    """
    Calcula los valores SHAP para el conjunto de características de prueba.

    Parameters:
        explainer (shap.Explainer): Objeto SHAP explainer.
        X_test (pd.DataFrame): Conjunto de características de prueba.

    Returns:
        shap_values: Valores SHAP calculados.
    """
    shap_values = explainer(X_test)
    return shap_values

def plot_shap_summary(
        shap_values, 
        X_test: pd.DataFrame,
        figures_path: Path
) -> None:
    """
    Genera y guarda un gráfico de resumen de los valores SHAP.

    Parameters:
        shap_values: Valores SHAP calculados.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        figures_path (Path): Ruta donde se guardará la figura.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot", fontsize=14)
    plt.tight_layout()
    output_path = figures_path / "shap_summary_forest.png"
    plt.savefig(output_path, dpi=300, bbox_inches = "tight")
    plt.close()

def plot_shap_dependence(
        shap_values, 
        X_test: pd.DataFrame,
        feature: str,
        figures_path: Path
) -> None:
    """
    Genera y guarda un gráfico de dependencia SHAP para una característica específica.

    Parameters:
        shap_values: Valores SHAP calculados.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        feature (str): Nombre de la característica para la que se generará el gráfico.
        figures_path (Path): Ruta donde se guardará la figura.
    """
    figures_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap_values_array = shap_values.values
    shap.dependence_plot(feature, shap_values_array, X_test, interaction_index=None, show=False)
    plt.title(f"SHAP Dependence Plot for {feature}", fontsize=14)
    plt.tight_layout()
    output_path = figures_path / f"shap_dependence_{feature}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # RUTAS
    processed_data_path = PROJECT_ROOT / "data" / "processed" / "dataset_final.csv"

    model_path = PROJECT_ROOT / "results" / "models" / "random_forest.pkl"


    print("ANÁLISIS SHAP - RANDOM FOREST")
    print("-" * 60)

    print("\nCargando modelo...")

    model = load_model(model_path)
    print("Modelo cargado correctamente")

    # Preparar conjuntos de test
    X_test, y_test = prepare_test_data(processed_data_path)
    print(f"Observaciones de test: {len(X_test)}")

    print(f"Número de variables: {X_test.shape[1]}")

    # Crear explainer SHAP
    explainer = create_shap_explainer(model)
    print("Explainer SHAP creado correctamente")

    # Calcular valores SHAP
    shap_values = calculate_shap_values(explainer, X_test)
    print("Valores SHAP calculados correctamente")
    print(f"Tipo de objetos SHAP: {type(shap_values)}")

    # Resultados
    print("SHAP COMPLETADO")
    print("-" * 60)

    figures_path = PROJECT_ROOT / "results" / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerando gráfico de resumen")
    plot_shap_summary(shap_values, X_test, figures_path)
    print(f"Gráfico de resumen guardado en: {figures_path / 'shap_summary_forest.png'}")

    print("\nGenerando gráfico de dependencia para la variable 'Gas_Price'")
    plot_shap_dependence(shap_values, X_test, feature="Gas_Price", figures_path=figures_path)
    print(f"Gráfico de dependencia guardado en: {figures_path / 'shap_dependence_Gas_Price.png'}")


if __name__ == "__main__":
    main()



