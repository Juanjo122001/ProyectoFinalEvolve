import os
import pandas as pd

def combined_data_mibgas(input_folder, output_file):

    # Definimos un DataFrame vacío para almacenar los datos combinados.
    combined_data = pd.DataFrame()

    # Creamos una función que nos permita iterar sobre la carpeta que contiene los archivos CSV y poder concatenarlos en un DataFrame.
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(input_folder, file)
            df = pd.read_csv(file_path, sep = ';', encoding = 'utf-8', skiprows = 1, quotechar='"')
            combined_data = pd.concat([combined_data, df], ignore_index=True)

    # Finalmente guardamos el DataFrame combinado en un archivo CSV.    
    combined_data.to_csv(output_file, sep = ';', index = False)

def main():
    # Definimos la carpeta de entrada y el archivo de salida
    input_folder = "data/raw/mibgas"
    output_file = "data/processed/mibgas_combined.csv"

    # Aseguramos que existe la carpeta de salida
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    
    # Llamamos a la función para combinar los datos
    combined_data_mibgas(input_folder, output_file)
    print(f"Datos combinados guardados en {output_file}")

if __name__ == "__main__":
    main()
    df = pd.read_csv("data/processed/mibgas_combined.csv", sep = ';', encoding = 'utf-8', skiprows = 1, quotechar='"')
    print(df.head())
            