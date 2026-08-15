from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Any
from genera_graficos import (
    plot_electricity_distribution,
    plot_gas_distribution,
    plot_europe_energy_evolution,
    plot_spain_vs_europe,
    plot_spain_vs_portugal,
    plot_gas_vs_electricity,
    plot_correlation_matrix,
    plot_heatmap_iberia,
    plot_top10_countries,
    plot_country_boxplot,
    plot_bottom10_countries
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "dataset_final.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT /
    "results" /
    "figures"
)

TABLES_DIR = (
    PROJECT_ROOT /
    "results" /
    "tables"
)

def create_output_directories() -> None:
    """
    Creamos directorios de salida para figuras y tablas
    """

    FIGURES_DIR.mkdir(
        parents = True,
        exist_ok =True
    )

    TABLES_DIR.mkdir(
        parents = True,
        exist_ok = True
    )


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Cargar el dataset procesado
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: \n{path}"
        )
    
    df = pd.read_csv(path, sep = ";")

    if df.empty:
        raise ValueError("El dataset está vacío")
    
    return df


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generar una vista general del dataset
    """
    overview = {
        "rows" : len(df),
        "columns" : len(df.columns),
        "countries" : df["Country"].nunique(),
        "start_semester": df["Semester"].min(),
        "end_semester" : df["Semester"].max(),
        "years" : sorted(df["Year"].unique().tolist()),
        "numerical_variables" : df.select_dtypes(
            include = "number"
        ).shape[1],
        "categorical_variables" : df.select_dtypes(
            exclude="number"
        ).shape[1],
        "missing_values" : int(df.isna().sum().sum()),
        "duplicate_rows" : int(df.duplicated().sum())
    }
    print(f"Filas: {overview['rows']:,}")
    print(f"Columnas: {overview['columns']}")
    print(f"Países: {overview['countries']}")
    print(
        f"Rango de tiempo: "
        f"{overview['start_semester']} → "
        f"{overview['end_semester']}"
    )
    print(f"Años: {overview['years']}")
    print(f"Variables numéricas: {overview['numerical_variables']}")
    print(f"Variables categóricas: {overview['categorical_variables']}")
    print(f"Valores faltantes: {overview['missing_values']}")
    print(f"Filas duplicadas: {overview['duplicate_rows']}")

    return overview

def save_dataset_overview(overview: dict) -> None:
    """
    Guardar el dataset overview a un archivo csv
    """

    output_path = (TABLES_DIR /
                   "dataset_overview.csv")

    overview_df = pd.DataFrame(
        {
            "Metric" : overview.keys(),
            "Value" : overview.values()
        }
    )

    overview_df.to_csv(
        output_path,
        index = False,
        encoding="utf-8-sig"
    )

    print(f"Dataset overview guardado correctamente como archivo .csv")

def generate_descriptive_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generar estadisticas descriptivas para todas las 
    variables numéricas
    """

    numeric = df.select_dtypes(include="number")

    statistics = pd.DataFrame(index=numeric.columns)

    statistics["Mean"] = numeric.mean()
    statistics["Median"] = numeric.median()
    statistics["Std"] = numeric.std()
    statistics["Min"] = numeric.min()
    statistics["Q1"] = numeric.quantile(0.25)
    statistics["Q3"] = numeric.quantile(0.75)
    statistics["Max"] = numeric.max()
    statistics["IQR"] = (
        statistics["Q3"]
        -
        statistics["Q1"]
    )
    statistics["CV"] = (
        statistics["Std"]
        /
        statistics["Mean"].abs()
    )
    statistics["Skewness"] = numeric.skew()
    statistics["Kurtosis"] = numeric.kurtosis()

    return statistics.round(4)

def save_descriptive_statistics(statistics : pd.DataFrame) -> None:
    """
    Función para guardar las estadísticas descriptivas
    """

    output = (
        TABLES_DIR / 
        "descriptive_statistics.csv"
    )

    statistics.to_csv(output, 
                      sep = ";",
                      index = True,
                      encoding="utf-8")

    print(f"Estadísticas descriptivas guardadas correctamente en un archivo .csv")
    


def generate_dataset_insights(df : pd.DataFrame) -> dict[str, Any]:
    """
    Función para generar conocimientos a partir
    de las variables numéricas
    """

    numeric = df.select_dtypes(include = "number")

    cv = (
        numeric.std() / numeric.mean().abs()
    )

    insights = {
        "highest_mean" : numeric.mean().idxmax(),
        "highest_std" :  numeric.std().idxmax(),
        "highest_cv" :  cv.idxmax(),
        "highest_skewness" :  numeric.skew().abs().idxmax(),
        "highest_kurtosis" : numeric.kurtosis().idxmax()
    }

    print("RESULTADOS OBTENIDOS")

    print(f"Mayor media: {insights['highest_mean']}")

    print(f"Mayor desviación estándar: {insights['highest_std']}")

    print(f"Mayor coeficiente de variación: {insights['highest_cv']}")

    print(f"Mayor Skewness: {insights['highest_skewness']}")

    print(f"Mayor kurtosis: {insights['highest_kurtosis']}")

    return insights

def save_dataset_insights(insights: dict[str, Any]) -> None:
    """
    Guardar resultados obtenidos del análisis anterior
    """

    output = (
        TABLES_DIR/
        "dataset_insights.csv"
    )

    (
        pd.Series(insights)
        .rename_axis("Metric")
        .reset_index(name = "Value")
        .to_csv(output, index = False, sep = ";", encoding="utf-8")
    )

def main():
    create_output_directories()

    dataset = load_dataset()

    overview = dataset_overview(dataset)
    save_dataset_overview(overview)

    statistics = generate_descriptive_statistics(dataset)
    save_descriptive_statistics(statistics)

    insights = generate_dataset_insights(dataset)
    save_dataset_insights(insights)

    print("\nGenerating visualizations...")

    plot_electricity_distribution(dataset)

    plot_gas_distribution(dataset)

    plot_europe_energy_evolution(dataset)

    plot_spain_vs_europe(dataset)

    plot_spain_vs_portugal(dataset)

    plot_gas_vs_electricity(dataset)

    plot_correlation_matrix(dataset)

    plot_heatmap_iberia(dataset)

    plot_top10_countries(dataset)

    plot_country_boxplot(dataset)

    plot_bottom10_countries(dataset)


if __name__ == "__main__":
    main()


