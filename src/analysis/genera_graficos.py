import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIGURES_DIR = (Path(__file__).resolve().parents[2]
/"results"
/"figures")

# Establecemos el mismo aspecto de todas las figuras
sns.set_theme(
    style = "whitegrid",
    context = "talk"
)

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# Establecemos una función para guardar figuras que sirva de forma general
def save_figure(filename: str) -> None:
    """
    La utilizamos para guardar la figura actual
    """
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        bbox_inches = "tight"
    )

    plt.close()


def plot_electricity_distribution(df: pd.DataFrame) -> None:
    """
    Graficar la distribución de electricidad 
    """
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="Electricity Price",
        bins=35,
        kde=True,
        stat="density",
        edgecolor="black",
        color = "steelblue",
        linewidth=0.5)

    plt.title(
        "Distribución de los Precios de la Electricidad (2021–2025)",
        fontweight="bold",
        fontsize = 14)

    plt.xlabel(
        "Precio de la Electricidad (€ / kWh)"
    )

    plt.ylabel(
        "Densidad"
    )

    plt.grid(
        axis = "y",
        linestyle = "--",
        alpha = 0.3
    )

    save_figure(
        "electricity_distribution.png"
    )
 


def plot_gas_distribution(
    df: pd.DataFrame,
) -> None:
    """
    Vamos a graficar la distribución de los precios
    del gas.
    """

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="Gas_Price",
        bins=35,
        kde=True,
        stat="density",
        color= "darkorange",
        edgecolor="black",
        linewidth=0.5)

    plt.title(
        "Distribución de los Precios del Gas (2021–2025)",
        fontweight="bold",
        fontsize = 14)

    plt.xlabel(
        "Distribución Precios del Gas (€ / kWh)"
    )

    plt.ylabel(
        "Densidad"
    )

    plt.grid(
        axis = "y",
        linestyle = "--",
        alpha = 0.3
    )

    save_figure(
        "gas_distribution.png"
    )


def plot_electricity_boxplot(df: pd.DataFrame) -> None:
    """
    Graficamos el boxplot de los precios 
    de la electricidad
    """

    plt.figure(figsize=(10,6))

    sns.boxplot(
        data = df,
        y = "Electricity Price",
        color = "steelblue",
        linewidth=1.2,
        fliersize = 3
    )

    plt.title("Boxplot de los Precios de la Electricidad (2021-2025)",
              fontsize = 14,
              fontweight = "bold")

    plt.ylabel("Precio de la Electricidad (€/kWh)")
    plt.xlabel("")

    plt.grid(
        axis = "y",
        linestyle = "--",
        alpha = 0.3
    )

    save_figure(
        "boxplot_electricity.png"
    )

def plot_gas_boxplot(df: pd.DataFrame) -> None:
    """
    Graficamos el boxplot de los precios 
    del gas
    """

    plt.figure(figsize=(10,6))

    sns.boxplot(
        data = df,
        y = "Gas_Price",
        color = "darkorange",
        linewidth=1.2,
        fliersize = 3
    )

    plt.title("Boxplot de los Precios del Gas (2021-2025)",
              fontsize = 14,
              fontweight = "bold")

    plt.ylabel("Precio del Gas (€/kWh)")
    plt.xlabel("")

    plt.grid(
        axis = "y",
        linestyle = "--",
        alpha = 0.3
    )

    save_figure(
        "boxplot_gas.png"
    )    


def plot_europe_energy_evolution(df:pd.DataFrame) -> None:
    """
    Graficamos la media de los precios de la electricidad en Europa y 
    el precio del gas del mercado ibérico, para ver la evolución de ambos a lo largo del tiempo.
    """

    europe = (
    df.groupby(
        ["Year", "Semester_Number", "Semester"],
        as_index=False
    )
    .agg(
        {"Electricity Price" : "mean",
        "Gas_Price" : "first"
        }
    )
    .sort_values(
        ["Year", "Semester_Number"]
    )
)

    plt.figure(figsize=(10,6))

    sns.lineplot(
        data = europe,
        x = "Semester",
        y = "Electricity Price",
        marker = "o",
        label = "Electricidad",
        color = "steelblue"
    )

    sns.lineplot(
        data = europe,
        x = "Semester",
        y = "Gas_Price",
        marker = "o",
        label = "Gas",
        color = "darkorange"
    )

    plt.title("Evolución del precio medio de la electricidad en Europa y del gas en el mercado ibérico (2021–2025)",
              fontsize = 14,
              fontweight = "bold")

    plt.ylabel("Precio (€/kWh)")
    plt.xlabel("Año")
    plt.xticks(rotation=45)

    plt.grid(
        axis = "y",
        linestyle = "--",
        alpha = 0.3
    )

    save_figure(
        "europe_average_prices.png"
    )


def plot_spain_vs_europe(df: pd.DataFrame) -> None:
    """
    Evolución del precio de la electricidad en España frente a la media de Europa.
    """

    europe = (
        df.groupby(
            ["Year", "Semester_Number", "Semester"],
            as_index=False
        )
        .agg(
            {
                "Electricity Price": "mean"
            }
        )
        .sort_values(
            ["Year", "Semester_Number"]
        )
    )

    spain = (
        df[df["Country"] == "Spain"]
        .groupby(
            ["Year", "Semester_Number", "Semester"],
            as_index=False
        )
        .agg(
            {
                "Electricity Price": "mean"
            }
        )
        .sort_values(
            ["Year", "Semester_Number"]
        )
    )

    plt.figure(figsize=(10,6))

    sns.lineplot(
        data=europe,
        x="Semester",
        y="Electricity Price",
        marker="o",
        linewidth=2.5,
        color="steelblue",
        label="Europe"
    )

    sns.lineplot(
        data=spain,
        x="Semester",
        y="Electricity Price",
        marker="o",
        linewidth=2.5,
        color="crimson",
        label="Spain"
    )

    plt.title(
        "Evolución del precio de la electricidad: España vs Europa (2021–2025)",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Semestre")

    plt.ylabel("Precio de la Electricidad (€/kWh)")

    plt.xticks(rotation=45)

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.legend()

    save_figure(
        "spain_vs_europe.png"
    )

def plot_spain_vs_portugal(df: pd.DataFrame) -> None:
    """
    Evolución del precio de la electricidad en España y Portugal.
    """

    iberia = (
        df[df["Country"].isin(["Spain", "Portugal"])]
        .groupby(
            ["Country", "Year", "Semester_Number", "Semester"],
            as_index=False
        )
        .agg(
            {
                "Electricity Price": "mean",
                "Gas_Price": "first"
            }
        )
        .sort_values(
            ["Country", "Year", "Semester_Number"]
        )
    )

    gas = (
        iberia[
            ["Year", "Semester_Number", "Semester", "Gas_Price"]
        ]
        .drop_duplicates()
        .sort_values(
            ["Year", "Semester_Number"]
        )
    )

    fig, ax1 = plt.subplots(figsize=(11, 6))

    sns.lineplot(
        data=iberia,
        x="Semester",
        y="Electricity Price",
        hue="Country",
        marker="o",
        linewidth=2.5,
        palette={
            "Spain": "crimson",
            "Portugal": "forestgreen"
        },
        ax=ax1
    )

    ax1.set_ylabel("Precio Electricidad (€/kWh)")
    ax1.set_xlabel("Semestre")

    ax2 = ax1.twinx()

    sns.lineplot(
        data=gas,
        x="Semester",
        y="Gas_Price",
        color="darkorange",
        marker="s",
        linewidth=2.5,
        label="MIBGAS",
        ax=ax2
    )

    ax2.set_ylabel("Precio del Gas (€/kWh)")

    ax1.set_title(
        "Evolución del precio de la electricidad en España y Portugal vs Precio del Gas Ibérico (2021–2025)",
        fontsize=14,
        fontweight="bold"
    )

    ax1.tick_params(axis="x", rotation=45)

    ax1.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        title="Series"
    )

    save_figure(
        "spain_vs_portugal.png"
    )

    
def plot_gas_vs_electricity(df: pd.DataFrame) -> None:
    """
    Visualizamos la relación entre los precios del gas y de la electricidad, 
    con un ajuste lineal para ver la tendencia.
    """

    plt.figure(figsize=(10, 6))

    df_scatter =  df[df["Country"].isin(["Spain", "Portugal"])].copy()

    sns.scatterplot(
        data=df_scatter,
        x="Gas_Price",
        y="Electricity Price",
        hue="Semester",
        palette="viridis",
        alpha=0.65,
        s=45
    )

    sns.regplot(
        data=df_scatter,
        x="Gas_Price",
        y="Electricity Price",
        scatter=False,
        color="black",
        line_kws={
            "linewidth": 2,
            "linestyle": "--"
        }
    )

    plt.title(
        "Relación entre los precios del gas y de la electricidad entre España y Portugal (2021–2025)",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Precio del Gas (€/kWh)")
    plt.ylabel("Precio de la Electricidad (€/kWh)")

    plt.grid(
        linestyle="--",
        alpha=0.3
    )

    plt.legend(
        title="Semester",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    save_figure(
        "gas_vs_electricity.png"
    )


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Visualizamos la matriz de correlación de Pearson
    para las principales variables numéricas.
    """

    numeric_columns = [
        "Electricity Price",
        "Semester_Number",
        "Ukraine_War",
        "Energy_Crisis",
        "Middle_East_Conflict"
    ]

    correlation = (
        df[numeric_columns]
        .corr(method="pearson", numeric_only=True)
    )

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Correlación de Pearson"}
    )

    plt.title(
        "Matriz de Correlación de Pearson en Europa (2021–2025)",
        fontsize=14,
        fontweight="bold"
    )

    save_figure(
        "correlation_matrix.png"
    )

def plot_heatmap_iberia(df: pd.DataFrame) -> None:
    """
    Genera la matriz de correlación entre el precio de la electricidad
    en España y Portugal y las principales variables energéticas
    y geopolíticas.
    """

    # ==========================================================
    # Filtramos únicamente España y Portugal
    # ==========================================================

    df_iberia = df[
        df["Country"].isin(["Spain", "Portugal"])
    ].copy()

    # ==========================================================
    # Agrupamos por semestre
    # ==========================================================

    df_iberia = (
        df_iberia
        .groupby(
            ["Year", "Semester_Number", "Semester"],
            as_index=False
        )
        .agg({
            "Electricity Price": "mean",
            "Gas_Price": "first",
            "Gas_Trend": "first",
            "Gas_Price_Pct_Change": "first",
            "Ukraine_War": "first",
            "Energy_Crisis": "first",
            "Middle_East_Conflict": "first"
        })
        .sort_values(
            ["Year", "Semester_Number"]
        )
    )

    # ==========================================================
    # Calculamos la matriz de correlación
    # ==========================================================

    correlation = (
        df_iberia[
            [
                "Electricity Price",
                "Gas_Price",
                "Gas_Trend",
                "Gas_Price_Pct_Change",
                "Ukraine_War",
                "Energy_Crisis",
                "Middle_East_Conflict"
            ]
        ]
        .corr(method="spearman")
    )

    # ==========================================================
    # Representación gráfica
    # ==========================================================

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="RdBu_r",
        center=0,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Coeficiente de correlación"}
    )

    plt.title(
        "Matriz de correlación (Spearman)\n"
        "España y Portugal (2021–2025)",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    save_figure(
        "heatmap_iberia.png"
    )

    plt.show()

def plot_top10_countries(df: pd.DataFrame) -> None:
    """
    Visualizamos los diez países europeos con los precios medios de electricidad más altos.
    """

    top10 = (
        df.groupby(
            "Country",
            as_index=False
        )
        .agg(
            {
                "Electricity Price": "mean"
            }
        )
        .sort_values(
            "Electricity Price",
            ascending=False
        )
        .head(10)
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=top10,
        x="Electricity Price",
        y="Country",
        color="firebrick"
    )

    plt.title(
        "Top 10 Países Europeos con los Precios Medios de Electricidad más Altos (2021–2025)",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Precio Medio (€/kWh)")
    plt.ylabel("País")

    plt.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )


    save_figure(
        "top10_electricity_prices.png"
    )

def plot_country_boxplot(df: pd.DataFrame) -> None:
    """
    Visualizamos la distribución de los precios de la electricidad
    para los quince países con los precios medios más altos.
    """

    top15 = (
        df.groupby(
            "Country",
            as_index=False
        )
        .agg(
            {
                "Electricity Price": "mean"
            }
        )
        .sort_values(
            "Electricity Price",
            ascending=False
        )
        .head(15)
    )

    top15_order = top15["Country"].tolist()

    boxplot_df = (
        df[df["Country"].isin(top15_order)]
        .copy()
    )

    plt.figure(figsize=(14, 7))

    sns.boxplot(
        data=boxplot_df,
        x="Country",
        y="Electricity Price",
        order=top15_order,
        palette="Blues",
        showfliers=False,
        linewidth=1
    )

    plt.title(
        "Distribución de los Precios de la Electricidad\nTop 15 Países Europeos por Precio Medio",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("País")
    plt.ylabel("Precio de la Electricidad (€/kWh)")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )


    save_figure(
        "country_boxplot_top15.png"
    )


def plot_bottom10_countries(df: pd.DataFrame) -> None:
    """
    Visualizamos los diez países europeos con los precios medios de electricidad más bajos.
    """

    bottom10 = (
        df.groupby(
            "Country",
            as_index=False
        )
        .agg(
            {
                "Electricity Price": "mean"
            }
        )
        .sort_values(
            "Electricity Price",
            ascending=True
        )
        .head(10)
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=bottom10,
        x="Electricity Price",
        y="Country",
        color="forestgreen"
    )

    plt.title(
        "Top 10 Países Europeos con los Precios Medios de Electricidad más Bajos (2021–2025)",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Precio Medio (€/kWh)")
    plt.ylabel("País")

    plt.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    save_figure(
        "bottom10_electricity_prices.png"
    )



