from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MIBGAS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mibgas"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gas_daily_prediction.csv"
)


# ============================================================
# CARGAR UN ARCHIVO MIBGAS
# ============================================================

def load_mibgas_file(file_path: Path) -> pd.DataFrame:
    """
    Carga un archivo CSV de MIBGAS y obtiene el precio diario
    del producto GDAES_D+1 en el PVB español.
    """

    df = pd.read_csv(
        file_path,
        sep=";",
        skiprows=1
    )

    # Limpiar nombres de columnas
    df.columns = df.columns.str.strip()

    # --------------------------------------------------------
    # Conversión de tipos
    # --------------------------------------------------------

    df["Trading day"] = pd.to_datetime(
        df["Trading day"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    df["MIBGAS Daily Price [EUR/MWh]"] = pd.to_numeric(
        df["MIBGAS Daily Price [EUR/MWh]"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Filtrar producto diario PVB España
    # --------------------------------------------------------

    df = df[
        (df["Product"] == "GDAES_D+1")
        & (df["Place of delivery"] == "PVB")
        & (df["Area"] == "ES")
    ].copy()

    # --------------------------------------------------------
    # Seleccionar columnas
    # --------------------------------------------------------

    df = df[
        [
            "Trading day",
            "MIBGAS Daily Price [EUR/MWh]"
        ]
    ]

    # Renombrar
    df = df.rename(
        columns={
            "Trading day": "Trading_Day",
            "MIBGAS Daily Price [EUR/MWh]": "Gas_Price"
        }
    )

    return df


# ============================================================
# CONSTRUIR SERIE MIBGAS
# ============================================================

def build_mibgas_daily_series() -> pd.DataFrame:
    """
    Carga todos los archivos MIBGAS disponibles y construye
    una única serie temporal diaria.
    """

    files = sorted(
        MIBGAS_PATH.glob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos CSV en: {MIBGAS_PATH}"
        )

    dataframes = []

    for file_path in files:

        print(
            f"Cargando: {file_path.name}"
        )

        df = load_mibgas_file(
            file_path
        )

        dataframes.append(df)

    # Unir todos los años
    dataset = pd.concat(
        dataframes,
        ignore_index=True
    )

    # Orden cronológico
    dataset = dataset.sort_values(
        "Trading_Day"
    )

    # Eliminar duplicados
    dataset = dataset.drop_duplicates(
        subset="Trading_Day",
        keep="first"
    )

    # Eliminar registros inválidos
    dataset = dataset.dropna(
        subset=[
            "Trading_Day",
            "Gas_Price"
        ]
    )

    dataset = dataset[
        dataset["Gas_Price"] > 0
    ]

    dataset = dataset.reset_index(
        drop=True
    )

    return dataset


# ============================================================
# VARIABLES TEMPORALES
# ============================================================

def create_temporal_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Crea variables temporales y cíclicas.
    """

    df = df.copy()

    df["Year"] = (
        df["Trading_Day"].dt.year
    )

    df["Month"] = (
        df["Trading_Day"].dt.month
    )

    df["Day"] = (
        df["Trading_Day"].dt.day
    )

    df["Day_of_Week"] = (
        df["Trading_Day"].dt.dayofweek
    )

    df["Day_of_Year"] = (
        df["Trading_Day"].dt.dayofyear
    )

    df["Quarter"] = (
        df["Trading_Day"].dt.quarter
    )

    df["Semester_Number"] = np.where(
        df["Month"] <= 6,
        1,
        2
    )

    # --------------------------------------------------------
    # Variables cíclicas
    # --------------------------------------------------------

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
# VARIABLES HISTÓRICAS
# ============================================================

def create_gas_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Crea variables históricas y estadísticas móviles.

    IMPORTANTE:
    Todas las variables utilizadas como predictores utilizan
    únicamente información disponible antes del día objetivo.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Lags
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Variación porcentual
    # --------------------------------------------------------

    df["Gas_Price_Pct_Change"] = (
        df["Gas_Price"]
        .pct_change()
        * 100
    )

    # Para predicción usamos el cambio conocido hasta ayer
    df["Gas_Price_Pct_Change_Lag_1"] = (
        df["Gas_Price_Pct_Change"].shift(1)
    )

    # --------------------------------------------------------
    # Tendencia
    # --------------------------------------------------------

    df["Gas_Trend"] = (
        df["Gas_Price_Lag_1"]
        - df["Gas_Price_Lag_7"]
    )

    # --------------------------------------------------------
    # Medias móviles
    # --------------------------------------------------------

    df["Gas_Rolling_Mean_7"] = (
        df["Gas_Price"]
        .shift(1)
        .rolling(
            window=7,
            min_periods=7
        )
        .mean()
    )

    df["Gas_Rolling_Mean_30"] = (
        df["Gas_Price"]
        .shift(1)
        .rolling(
            window=30,
            min_periods=30
        )
        .mean()
    )

    # --------------------------------------------------------
    # Volatilidad
    # --------------------------------------------------------

    df["Gas_Rolling_Std_7"] = (
        df["Gas_Price"]
        .shift(1)
        .rolling(
            window=7,
            min_periods=7
        )
        .std()
    )

    return df


# ============================================================
# VARIABLES GEOPOLÍTICAS
# ============================================================

def create_geopolitical_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Añade las variables asociadas a acontecimientos
    geopolíticos utilizados en el TFM.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Guerra de Ucrania
    # --------------------------------------------------------

    df["Ukraine_War"] = (
        df["Trading_Day"]
        >= pd.Timestamp("2022-02-24")
    ).astype(int)

    # --------------------------------------------------------
    # Crisis energética
    # --------------------------------------------------------

    df["Energy_Crisis"] = (
        (df["Trading_Day"] >= pd.Timestamp("2022-01-01"))
        &
        (df["Trading_Day"] <= pd.Timestamp("2023-12-31"))
    ).astype(int)

    # --------------------------------------------------------
    # Conflicto de Oriente Medio
    # --------------------------------------------------------

    df["Middle_East_Conflict"] = (
        df["Trading_Day"]
        >= pd.Timestamp("2023-10-07")
    ).astype(int)

    return df


# ============================================================
# LIMPIEZA FINAL
# ============================================================

def clean_final_dataset(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Realiza la limpieza final y elimina las filas iniciales
    que no disponen de suficiente historial para calcular
    los lags y estadísticas móviles.
    """

    df = df.copy()

    required_columns = [
        "Gas_Price_Lag_1",
        "Gas_Price_Lag_7",
        "Gas_Price_Lag_14",
        "Gas_Price_Lag_30",
        "Gas_Price_Pct_Change_Lag_1",
        "Gas_Rolling_Mean_7",
        "Gas_Rolling_Mean_30",
        "Gas_Rolling_Std_7"
    ]

    df = df.dropna(
        subset=required_columns
    )

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# DATASET COMPLETO
# ============================================================

def build_gas_daily_dataset() -> pd.DataFrame:
    """
    Ejecuta todo el pipeline de preparación del dataset diario.
    """

    print("\nConstruyendo serie diaria MIBGAS...")

    df = build_mibgas_daily_series()

    print(
        f"Registros después de cargar MIBGAS: "
        f"{len(df)}"
    )

    print("\nCreando variables temporales...")

    df = create_temporal_features(
        df
    )

    print("Creando variables históricas...")

    df = create_gas_features(
        df
    )

    print("Creando variables geopolíticas...")

    df = create_geopolitical_features(
        df
    )

    print("Realizando limpieza final...")

    df = clean_final_dataset(
        df
    )

    return df


def validate_dataset(
    df: pd.DataFrame
) -> None:
    """
    Muestra información básica para comprobar que el dataset
    se ha construido correctamente.
    """

    print("\n" + "=" * 70)
    print("VALIDACIÓN DATASET DIARIO DE GAS")
    print("=" * 70)

    print(
        f"\nNúmero de registros: {len(df)}"
    )

    print(
        f"Número de variables: {len(df.columns)}"
    )

    print(
        f"\nPeriodo:"
        f"\n{df['Trading_Day'].min().date()}"
        f" → "
        f"{df['Trading_Day'].max().date()}"
    )

    print(
        f"\nDuplicados:"
        f" {df['Trading_Day'].duplicated().sum()}"
    )

    print(
        f"Nulos:"
        f" {df.isna().sum().sum()}"
    )

    print(
        f"\nPrecio mínimo:"
        f" {df['Gas_Price'].min():.2f} EUR/MWh"
    )

    print(
        f"Precio máximo:"
        f" {df['Gas_Price'].max():.2f} EUR/MWh"
    )

    print(
        f"Precio medio:"
        f" {df['Gas_Price'].mean():.2f} EUR/MWh"
    )

    print("\nColumnas:")
    for column in df.columns:
        print(f"  - {column}")




def main():

    print("=" * 70)
    print("PREPARACIÓN DATASET DIARIO MIBGAS")
    print("=" * 70)

    df = build_gas_daily_dataset()

    validate_dataset(
        df
    )

    # Crear directorio de salida
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Guardar dataset
    df.to_csv(
        OUTPUT_PATH,
        sep=";",
        index=False
    )

    print(
        "\nDataset guardado correctamente en:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()