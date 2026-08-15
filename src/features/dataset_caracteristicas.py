# Importamos las librerías necesarias
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

ELECTRICITY_DATASET_PATH = (
    INTERIM_DIR / "clean_electricity_prices.csv"
)

MIBGAS_DATASET_PATH = (
    INTERIM_DIR / "mibgas_combined.csv"
)

OUTPUT_DATASET_PATH = (
    PROCESSED_DIR / "dataset_final.csv"
)

OUTPUT_SUMMARY_PATH = (
    PROCESSED_DIR / "semester_summary.csv"
)

# Leemos los datasets que hemos creado previamente
dataset_electricidad_europa = pd.read_csv("data/processed/clean_electricity_prices.csv", sep = ';')
dataset_mibgas = pd.read_csv("data/processed/mibgas_combined.csv", sep = ';')

# Verificación de que los datasets se han cargado correctamente
print(dataset_electricidad_europa.head())
print()
print(dataset_mibgas.head())
print()

def prepare_electricity_dataset(df):
    """
    Función que prepara el dataset Europeo para su posterior
    integración con el de MIBGAS
    """

    # Vamos a trabajar sobre una copia
    df = df.copy()

    # Procedemos a ordenar cronológicamente los registros
    df = df.sort_values(
        by = ["Country", "Semester"]
    ).reset_index(drop=True)

    # Nos aseguramos que el precio sea numérico
    df["Electricity Price"] = pd.to_numeric(
        df["Electricity Price"],
        errors = "coerce"
    )

    # Observamos el número de duplicados por si en un futuro se añaden nuevos registros
    duplicates = df.duplicated(
        subset = [
            "Country",
            "Semester",
            "consumption_band",
            "tax_type",
            "currency"
        ]
    )

    print(f"Duplicados en electricidad: {duplicates}")

    return df


def prepare_mibgas_dataset(df):
    """
    Prepara el dataset de mibgas para integrarlo con 
    el dataset europeo
    """

    df = df.copy()

    # Nos vamos a quedar con el mercado diario del gas para poder hacer transformaciones y comparar ambos datasets
    df = df[df["Product"] == "GDAES_D+1"].copy()

    # Convertimos la fecha a formato datetime
    df["Trading day"] = pd.to_datetime(
        df["Trading day"],
        dayfirst=True
    )

    # Extraemos año y mes
    df["Year"] = df["Trading day"].dt.year
    df["Month"] = df["Trading day"].dt.month


    # Creamos el semestre para trabajar en la misma franja temporal que el dataset europeo
    df["Semester"] = np.where(
        df["Month"] <= 6,
        df["Year"].astype(str) + "-S1",
        df["Year"].astype(str) + "-S2"
    )


    # Agrupamos por semestre y extraemos las estadísticas más interesantes para nuestro análisis
    df = df.groupby("Semester").agg(
        Gas_Price = ("MIBGAS Daily Price [EUR/MWh]", "mean"),
        Gas_Price_Median = ("MIBGAS Daily Price [EUR/MWh]", "median"),
        Gas_Price_Max = ("MIBGAS Daily Price [EUR/MWh]", "max"),
        Gas_Price_Min = ("MIBGAS Daily Price [EUR/MWh]", "min"),
        Gas_Price_Std = ("MIBGAS Daily Price [EUR/MWh]", "std"),
        Gas_Price_Range = ("MIBGAS Daily Price [EUR/MWh]", lambda x: x.max() - x.min()), # Rango de Precios
        Gas_Price_CV = ("MIBGAS Daily Price [EUR/MWh]", lambda x: x.std() / x.mean()) # Coeficiente de Variación: Variabilidad de precios en relación con nivel medio
    ).reset_index()

    # Convertimos todas las variables de precio de €/MWh a €/kWh
    price_columns = [
        column
        for column in df.columns
        if column.startswith("Gas_Price")
    ]
    df[price_columns] = df[price_columns] / 1000

    return df

def merge_datasets(df1, df2):
    """
    Esta función va a ser creada para unir los dos datasets en uno solo
    """
    df_merge = df1.merge(
        df2,
        on = "Semester",
        how = "left"
    )

    return df_merge

def add_temporal_features(df):
    """
    Añade variables temporales que vienen derivadas
    de la columna Semestre
    """
    df = df.copy()

    df["Year"] = (
        df["Semester"].str.extract(r"(\d{4})").astype(int) 
    )

    df["Semester_Number"] = (
        df["Semester"].str.extract(r"S(\d)").astype(int)
    )

    return df

def add_geopolitical_features(df):
    """
    Función para añadir los sucesos geopolíticos que han
    marcado las variaciones más relevantes en los precios.
    """

    df = df.copy()

    # Invasión Rusa de Ucrania
    df["Ukraine_War"] = (df["Year"]>= 2022).astype(int)

    # Crsis Energética Europea
    df["Energy_Crisis"] = (
        (
            (df["Year"] == 2022) |
            (df["Year"] == 2023)
        )
    ).astype(int)

    # Conflictos Iran-Israel
    df["Middle_East_Conflict"] = (
         (
            (df["Year"] == 2024) &
            (df["Semester_Number"] == 1)
        ) |
        (df["Year"] >= 2025)
    ).astype(int)

    return df

def add_growth_features(df):
    """
    Función que permite detectar los cambios de precio en el gas,
    las tendencias y los cambios porcentuales
    """

    df = df.copy()

    # Obtenemos un unico valor del gas
    gas = (
        df[["Semester", "Year", "Semester_Number", "Gas_Price"]]
        .drop_duplicates()
        .sort_values(["Year", "Semester_Number"])
    )

    #Calculamos la variación absoluta semestre a semestre
    gas["Gas_Price_Change"] = gas["Gas_Price"].diff()

    #Calculamos la variación porcentual semstre a semestre
    gas["Gas_Price_Pct_Change"] = gas["Gas_Price"].pct_change()

    #Calculamos la tendencia del precio del gas
    gas["Gas_Trend"] = (
        np.sign(gas["Gas_Price_Change"])
        .fillna(0)
        .astype(int)
    )

    # Unimos las nuevas características al dataset original
    df = df.merge(
        gas[
            [
            "Semester",
            "Gas_Price_Change",
            "Gas_Price_Pct_Change",
            "Gas_Trend",
        ]
        ],
        on = "Semester",
        how = "left"
    )

    return df

def create_semester_summary(df):
    """
    Crea un dataset resumen a nivel de semestre para el análisis exploratorio.
    
    El dataset resultante contiene estadísticas descriptivas de los precios de la electricidad, junto con las estadísticas del mercado de gas y variables geopolíticas para cada semestre.
    """
    electricity_summary = (
        df.groupby(
            ["Semester", "Year", "Semester_Number"],
            as_index = False
        )
        .agg(
            Country = ("Country", "nunique"),
            Observations = ("Country", "count"),
            Avg_Electricity_Price=("Electricity Price", "mean"),
            Median_Electricity_Price = ("Electricity Price", "median"),
            Std_Electricity_Price = ("Electricity Price", "std"),
            Min_Electricity_Price = ("Electricity Price", "min"),
            Max_Electricity_Price = ("Electricity Price", "max"),
            Q1_Electricity_Price=("Electricity Price", lambda x: x.quantile(0.25)),
            Q3_Electricity_Price=("Electricity Price", lambda x: x.quantile(0.75))
        )
    )

    gas_summary = (
        df[
            [  "Semester",
                "Gas_Price",
                "Gas_Price_Min",
                "Gas_Price_Max",
                "Gas_Price_Median",
                "Gas_Price_Std",
                "Gas_Price_Range",
                "Gas_Price_CV",
                "Gas_Price_Change",
                "Gas_Price_Pct_Change",
                "Gas_Trend",
                "Ukraine_War",
                "Energy_Crisis",
                "Middle_East_Conflict"
             ]
        ]
        .drop_duplicates(subset = "Semester")
    )

    semester_summary = electricity_summary.merge(
        gas_summary,
        on = "Semester",
        how = "left"
    )

    semester_summary = semester_summary.sort_values(
        ["Year", "Semester_Number"]
    ).reset_index(drop = True)

    return semester_summary

def save_dataset(df: pd.DataFrame, output_path: Path) -> None:

    """
    Guardamos el dataset final creado
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        sep = ";"
    )

    print(f"Dataset guardado correctamente: {output_path}")


def save_semester_summary(summary: pd.DataFrame, output_path:Path) -> None:
    """
    Guardar el resumen del semestre como dataset
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        output_path,
        index = False,
        encoding="utf-8",
        sep = ";"
    )

    print(f"Resumen Semestral ha sido guardado de forma exitosa: {output_path}")

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reordenar las columnas del dataset
    """
    priority_columns = [
        "Country",
        "Semester",
        "Year",
        "Semester_Number",
        "Electricity Price",
        "Gas_Price",
        "Gas_Price_Change",
        "Gas_Price_Pct_Change",
        "Gas_Trend",
        "Ukraine_War",
        "Energy_Crisis",
        "Middle_East_Conflict"
    ]

    remaining_columns = [col for col in df.columns if col not in priority_columns]

    return df[priority_columns + remaining_columns]


def validate_dataset_final(df):
    """
    El objetivo es realizar una validación final del dataset para verificar
    que está preparado para el análisis exploratorio y el entrenamiento de modelos.
    """

    print("VALIDACIÓN FINAL DATASET")
    print('-'*70)

    # Empezamos con el número de registros
    print(f"\nNúmero de registros: {len(df):,}")
    print(f"Número de variables: {len(df.columns)}")

    # Rango de tiempo de nuestro dataset
    print("\nRango de tiempo abarcado")
    print(f"Primer Semestre: {df['Semester'].min()}")
    print(f"Último Semestre: {df['Semester'].max()}")

    print(f"Años abarcados: {sorted(df['Year'].unique())}")

    # Número de países
    print("\nNúmero de países:")
    print(df["Country"].nunique())

    # Valores nulos presentes
    nulls = df.isnull().sum()
    print(F"\nValores nulos: {nulls}")

    # Valores duplicados presentes
    duplicates = df.duplicated(
        subset = ["Country",
                  "Semester",
                  "consumption_band",
                  "tax_type",
                  "currency"]
    ).sum()
    print(f"\nValores duplicados: {duplicates}")

    # Tipos de datos presentes
    print("\nTipos de datos:")
    print(df.dtypes)

    # Variables numéricas
    print("\nResumen estadístico:")
    print(
        df[
            [
                "Electricity Price",
                "Gas_Price"
            ]
        ].describe()
    )

    print("\n" + '-' * 70)

    # Comprobaciones finales
    if df.empty:
        raise ValueError("El dataset está vacío")
    

    ALLOWED_NULL_COLUMNS = [
    "Gas_Price_Change",
    "Gas_Price_Pct_Change"]

    nulls = df.drop(
    columns=ALLOWED_NULL_COLUMNS
    ).isna().sum()

    nulls = nulls[nulls > 0]

    if len(nulls) > 0:
        raise ValueError(
            f"Valores faltantes inesperados:\n{nulls}"
    )


    if duplicates > 0:
        raise ValueError(
            f"Se han encontrado {duplicates} registros duplicados"
        )
    
    negative_prices = df[df["Electricity Price"] < 0]

    if not negative_prices.empty:
        print(
            f"AVISO: Se han encontrado {len(negative_prices)} "
            "precios negativos procedentes del dataset original de Eurostat."
        )
    
    if(df["Gas_Price"] < 0).any():
        raise ValueError(
            "Existen precios negativos del gas"
        )
    
    years = set(df["Year"])
    expected_years = {2021, 2022, 2023, 2024, 2025}
    if not years.issubset(expected_years):
        raise ValueError(
            f"Años inesperados encontrados: {years - expected_years}"
        )
    
    semesters = set(df["Semester_Number"])
    if semesters != {1, 2}:
        raise ValueError(
            "La columna Semester_Number solo puede contener 1 o 2"
        )
    
    print("\n Todas las comprobaciones han sido superadas satisfactoriamente")

    return True


def main():
    print("Preparando dataset final")

    electricity = prepare_electricity_dataset(dataset_electricidad_europa)

    gas = prepare_mibgas_dataset(dataset_mibgas)

    dataset = merge_datasets(electricity, gas)

    dataset = dataset[
    (dataset["consumption_band"] == "DC") &
    (dataset["currency"] == "Euro")
].copy()

    dataset = add_temporal_features(dataset)
    
    dataset = add_geopolitical_features(dataset)

    dataset = add_growth_features(dataset)

    dataset = reorder_columns(dataset)

    validate_dataset_final(dataset)

    semester_summary = create_semester_summary(dataset)

    save_dataset(dataset, OUTPUT_DATASET_PATH)

    save_semester_summary(semester_summary, OUTPUT_SUMMARY_PATH)
    
    print("\nDataset final generado correctamente")


if __name__ == "__main__":
    main()




