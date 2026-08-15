import os
import pandas as pd
import numpy as np
from pathlib import Path

# Definimos la ruta base del proyecto, el archivo de entrada y el archivo de salida
BASE_DIR = Path(__file__).resolve().parents[2] # Directorio base del proyecto

INPUT_FILE = (
    BASE_DIR 
    /"data" 
    /"raw"
    /"raw_data.xlsx"
)

OUTPUT_FILE = (
    BASE_DIR
    /"data"
    /"processed"
    /"clean_electricity_prices.csv"
)

# Creamos un diccionario que nos permite traducir los nombres que nos proporciona nuestro archivo a nombres más cómodos para trabajar
METADATA_MAPPING = {
    "Time frequency": "time_frequency",
    "Standard international energy product classification (SIEC)" : "energy_product",
    "Energy consumption": "energy_consumption",
    "Unit of measure": "unit_of_measure",
    "Taxes": "tax_type",
    "Currency": "currency",
    }


# Definimos una función para limpiar los precios
def clean_eurostat_value(value):

    """
    Función para limpiar los valores de precios de electricidad.
    Convierte los valores a float y maneja los valores faltantes.
    """
    # Comprobamos si el valor es un dato faltante
    if pd.isna(value):
        return np.nan
    
    # Convertimos a string y eliminamos espacios
    value = str(value).strip()

    # Elimina símbolos especiales que vienen en nuestro dataset y los reemplaza por NaN
    if value in [":", "", None]:
        return np.nan
    
    # Reemplazamos estas letras que aparecen en el dataset junto a los valores numéricos y que no las necesitamos para el análisis
    for letter in ["e", "p", "b", "c"]:
        value = value.replace(letter, "")

    value = value.strip()

    try:
        return float(value)
    except ValueError:
        return np.nan


def extract_metadata(sheet_name):
     """
     Extrae los metadatos de una hoja del Excel de Eurostat.
     
     Cada hoja contiene, en sus primeras filas, información descriptiva
     sobre los datos (frecuencia temporal, banda de consumo, impuestos,
     moneda, etc.). Esta función lee únicamente esas filas y devuelve
     un diccionario con los metadatos que utilizaremos posteriormente
     para enriquecer el dataset final.

     Parameters
     ----------
     sheet_name : str
     Nombre de la hoja del Excel que se desea procesar.

     Returns
     -------
     dict
     Diccionario con los metadatos de la hoja.
     """

     # Leemos los metadatos de las 10 primeras filas de las hojas de nuestro archivo de Excel
     df_metadata = pd.read_excel(
         INPUT_FILE, 
         sheet_name = sheet_name,
         nrows = 10,
         header = None
     )

     # Creamos un diccionario vacío para almacenar los metadatos
     metadata = {}

     # Recorremos todas las filas del bloque de metadatos
     for _, row in df_metadata.iterrows():
         # Obtenemos la clave (título en negrita de la fila) y el valor (contenido de la fila)
         key = row.iloc[0]
         if pd.isna(key): # Si no hay clave continuamos
             continue
         
         key = str(key).strip()

         value = None

         for cell in row.iloc[1:]:
            if pd.notna(cell):
                value = str(cell).strip()
                break

         # Si no encontramos ningún valor, pasamos a la siguiente fila
         if value is None:
             continue

         # Solo vamos a almacenar aquellos campos que estén en nuestro diccionario de mapeo
         if key in METADATA_MAPPING:
             metadata[METADATA_MAPPING[key]] = value
     
     return metadata


# Procedemos a crear una función para limpiar el dataset y quedarnos con las filas que nos interesan realmente
def process_sheet(sheet_name):
    """
    Procesa una hoja de Excel y devuelve
    un DataFrame limpio con los datos de interés.
    """

    metadata = extract_metadata(sheet_name)

    # Leemos nuestra hoja de Excel y la almacenamos en un DataFrame
    df = pd.read_excel(
        INPUT_FILE, 
        sheet_name = sheet_name,
        header = 11 # Fila exacta en la que se encuentran los nombres de las columnas, fila 12.
        )
    
    # Hacemos una pequeña comprobación para ver que se está leyendo y cargando correctamente
    # print(f"\n Procesando hoja: {sheet_name}")

    # Eliminamos la linea de GEO (Labels) que aparece
    df = df[df["TIME"] != "GEO (Labels)"]

    # Renombramos la columna principal
    df.rename(
        columns = {"TIME": "Country"}, 
        inplace = True
        )
    
    # Eliminamos las columnas vacías como medida de seguridad
    df.dropna(
        axis = 1,
        how = "all", # Elimina las columnas completamente vacías
        inplace = True
    )

    # Transformamos el DataFrame a formato largo
    df = df.melt(
         id_vars = "Country",
         var_name = "Semester",
         value_name = "Electricity Price"
        )

    # Limpiamos los precios con la función creada previamente
    df["Electricity Price"] = df["Electricity Price"].apply(clean_eurostat_value)

    # Añadimos los metadatos
    for column, value in metadata.items():
        df[column] = value
    
    # Conservamos la descripción completa.
    df["consumption_description"] = df["energy_consumption"]

    # Extraemos únicamente el código DA, DB, DC que nos será útil para el análisis
    df["consumption_band"] = (
        df["energy_consumption"]
        .str.extract(r'band\s+([A-Z]+)$')
    )
    
    # Eliminamos las filas con precio desconocido
    df.dropna(
        subset = ["Electricity Price"],
        inplace = True
    )

    return df

def build_dataset():
    """
    Procesa todas las hojas del archivo Excel y construye
    un único DataFrame con toda la información.

    Returns
    -------
    pandas.DataFrame
        Dataset completo con todas las hojas unidas.
    """

    # Primero abrimos el archivo excel para conocer las hojas que tenemos disponibles
    archivo = pd.ExcelFile(INPUT_FILE)

    # Vamos a eliminar la hoja Summary, ya que únicamente tiene información descriptiva del archivo
    sheet_names = [sheet for sheet in archivo.sheet_names if sheet != "Summary"]

    # Creamos también una lista donde se almacenará temporalmente el DataFrame obtenido por cada hoja
    dataframes = []

    # Procesamos todas las hojas
    for sheet in sheet_names:
        print(f"Procesando hoja: {sheet}")
        df_sheet = process_sheet(sheet)
        dataframes.append(df_sheet)

    # Unimos todos los dataframes en un único dataframe
    df_final = pd.concat(
        dataframes,
        ignore_index= True
    )
    return df_final

def remove_aggregates(df):
    """
    Elimina los agregados europeos y otras regiones que no representan
    países individuales.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset completo.

    Returns
    -------
    pandas.DataFrame
        Dataset únicamente con países.
    """

    aggregate_keywords = [
        "European Union",
        "Euro area",
        "EA",
        "EU",
        "countries",
        "candidate",
        "euro area"
    ]

    # Creamos una máscara que identifica las filas que contienen las palabras anteriores
    mask = df["Country"].str.contains(
        "|".join(aggregate_keywords),
        case = False,
        na = False
    )

    # Conservamos únicamente las filas que no pertenecen a los agregados
    return df[~mask].copy()

def validate_dataset(df):
    """
    Realiza una auditoría básica del dataset generado para comprobar
    que el proceso ETL se ha ejecutado correctamente.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset completo.

    Returns
    -------
    None
    """

    print("\n" + "=" * 60)
    print("VALIDACIÓN DEL DATASET")
    print("=" * 60)

    # ==============================================================
    # Dimensiones
    # ==============================================================

    print(f"\nNúmero de filas: {len(df):,}")
    print(f"Número de columnas: {len(df.columns)}")

    # ==============================================================
    # Países
    # ==============================================================

    print(f"\nNúmero de países: {df['Country'].nunique()}")

    # ==============================================================
    # Cobertura temporal
    # ==============================================================

    print(f"\nPrimer semestre: {df['Semester'].min()}")
    print(f"Último semestre: {df['Semester'].max()}")

    # ==============================================================
    # Valores nulos
    # ==============================================================

    print("\nValores nulos por columna:")
    print(df.isna().sum())

    # ==============================================================
    # Duplicados
    # ==============================================================

    duplicated = df.duplicated().sum()
    print(f"\nRegistros duplicados: {duplicated}")

    # ==============================================================
    # Bandas de consumo
    # ==============================================================

    print("\nBandas de consumo:")
    print(sorted(df["consumption_band"].unique()))

    # ==============================================================
    # Tipos de impuestos
    # ==============================================================

    print("\nTipos de impuestos:")
    print(df["tax_type"].unique())

    # ==============================================================
    # Monedas
    # ==============================================================

    print("\nMonedas:")
    print(df["currency"].unique())
    print("\n" + "=" * 60)


def main():
    """Función principal de nuestro proceso de ETL"""
    print("\nConstruyendo nuestro dataset...")

    # Construimos el dataset completo
    df_electricidad = build_dataset()

    # Eliminamos los agregados
    df_electricidad = remove_aggregates(df_electricidad)

    # Validamos los resultados
    validate_dataset(df_electricidad)

    # Guardamos el dataset en un archivo .csv
    df_electricidad.to_csv(
        OUTPUT_FILE,
        sep = ";",
        index = False
    )

    print("\nDataset guardado correctamente")

if __name__ == "__main__":
    main()



    

    


    


