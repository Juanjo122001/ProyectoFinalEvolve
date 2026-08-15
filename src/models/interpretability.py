import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from preprocessing import prepare_ml_dataset
from dataset_split import temporal_train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_feature_importance(model, feature_names):
    """Obtiene la importancia de variables de distintos modelos.
    Modelos soportados
    ------------------
    - Random Forest
    - Optimized Random Forest
    - XGBoost
    - LightGBM
    - CatBoost
    - Linear Regression
    - Ridge Regression """

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance": model.feature_importances_

        })

    elif hasattr(model, "coef_"):

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance": abs(model.coef_),

            "Coefficient": model.coef_

        })

    else:

        raise ValueError(
            f"{type(model).__name__} no soporta extracción de importancia."
        )

    importance = (
        importance
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importance


def save_feature_importance(importance_df, output_path):
    """
    Guarda la importancia de las variables en un archivo CSV.

    Parameters:
    - importance_df: DataFrame
        DataFrame con la importancia de las variables.
    - output_path: str
        Ruta donde se guardará el archivo CSV.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    importance_df.to_csv(output_path, sep = ";", index=False)

    print()

    print(f"Importancia de las variables guardada en: {output_path}")


def plot_feature_importance(importance_df: pd.DataFrame, 
                            output_path: str | Path,
                            model_name: str,
                            top_n: int = 15):
    """
    Genera un gráfico de barras con la importancia de las variables.

    Parameters:
    - importance_df: DataFrame
        DataFrame con la importancia de las variables.
    - output_path: str | Path
        Ruta donde se guardará el gráfico.
    - top_n: int
        Número de variables más importantes a mostrar.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Determinar variable a representar
    # ------------------------------------------------------

    if "Coefficient" in importance_df.columns:

        value_column = "Coefficient"

        plot_df = (
            importance_df
            .copy()
            .sort_values(
                by="Importance",
                ascending=False
            )
            .head(top_n)
            .sort_values(
                by="Coefficient",
                ascending=True
            )
        )

        xlabel = "Coeficiente"

    else:

        value_column = "Importance"

        plot_df = (
            importance_df
            .head(top_n)
            .sort_values(
                by="Importance",
                ascending=True
            )
        )

        xlabel = "Importancia"

    # ------------------------------------------------------
    # Crear figura
    # ------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_df["Feature"],
        plot_df[value_column]
    )

    plt.xlabel(xlabel)

    plt.ylabel("Variable")

    plt.title(
        f"{model_name} - Importancia de variables"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Figura guardada en:\n{output_path}"
    )

def run_feature_importance(
    model,
    model_name: str,
    feature_names
) -> pd.DataFrame:
    """
    Ejecuta el análisis completo de importancia para un modelo.

    Obtiene las importancias, guarda el CSV y genera la figura.

    Parameters
    ----------
    model : object
        Modelo entrenado.

    model_name : str
        Nombre del modelo.

    feature_names : iterable
        Nombres de las variables utilizadas.

    Returns
    -------
    pd.DataFrame
        DataFrame con las importancias calculadas.
    """

    importance = get_feature_importance(
        model,
        feature_names
    )

    # ------------------------------------------------------
    # Nombre seguro para los archivos
    # ------------------------------------------------------

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    # ------------------------------------------------------
    # Guardar tabla
    # ------------------------------------------------------

    table_path = (
        PROJECT_ROOT
        / "results"
        / "tables"
        / f"{filename}_feature_importance.csv"
    )

    save_feature_importance(
        importance,
        table_path
    )

    # ------------------------------------------------------
    # Guardar figura
    # ------------------------------------------------------

    figure_path = (
        PROJECT_ROOT
        / "results"
        / "figures"
        / f"{filename}_feature_importance.png"
    )

    plot_feature_importance(
        importance,
        figure_path,
        model_name
    )

    return importance

def main():
    """
    Ejecuta el análisis de interpretabilidad de todos los modelos.
    """
    print("ANÁLISIS DE INTERPRETABILIDAD DE LOS MODELOS")
    print("-" * 60)

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    processed_data_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "dataset_final.csv"
    )

    X, y = prepare_ml_dataset(
        processed_data_path
    )

    X_train, X_test, y_train, y_test = (
        temporal_train_test_split(
            X,
            y
        )
    )

    feature_names = X_train.columns

    # ------------------------------------------------------
    # Modelos disponibles
    # ------------------------------------------------------

    models = {

        "Random Forest":
            "random_forest.pkl",

        "Optimized Random Forest":
            "optimized_random_forest.pkl",

        "XGBoost":
            "xgboost.pkl",

        "LightGBM":
            "lightgbm.pkl",

        "CatBoost":
            "catboost.pkl",

        "Ridge Regression":
            "ridge_regression.pkl",

        "Linear Regression":
            "linear_regression.pkl"

    }

    # ------------------------------------------------------
    # Analizar modelos
    # ------------------------------------------------------

    for model_name, filename in models.items():

        model_path = (
            PROJECT_ROOT
            / "results"
            / "models"
            / filename
        )

        print()
        print(model_name)
        print("-" * 60)

        # Comprobar que existe el modelo

        if not model_path.exists():

            print(
                f"ADVERTENCIA: no se encuentra el modelo:\n"
                f"{model_path}"
            )

            continue

        # Cargar modelo

        model = joblib.load(
            model_path
        )

        # Obtener y guardar importancia

        importance = run_feature_importance(
            model,
            model_name,
            feature_names
        )

        # Mostrar Top 10

        print()
        print("TOP 10 VARIABLES")
        print("-" * 60)

        print(
            importance.head(10).to_string(
                index=False
            )
        )

    print()
    print("ANÁLISIS COMPLETADO")
    print("-" * 60)



if __name__ == "__main__":
    main()