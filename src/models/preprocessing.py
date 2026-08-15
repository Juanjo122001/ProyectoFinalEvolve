import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "Country",
    "Gas_Price",
    "Gas_Price_Pct_Change",
    "Gas_Trend",
    "Ukraine_War",
    "Energy_Crisis",
    "Middle_East_Conflict",
    "Year",
    "Semester_Number",
]

TARGET = "Electricity Price"


def load_dataset(filepath: str | Path) -> pd.DataFrame:
    """
    Carga el dataset preparado para Machine Learning.

    Parameters
    ----------
    filepath : str | Path
        Ruta del dataset final.

    Returns
    -------
    pd.DataFrame
        Dataset cargado.
    """

    return pd.read_csv(filepath, sep = ";")

def select_features(
    df: pd.DataFrame,
    features: list[str],
    target: str
) -> pd.DataFrame:
    """
    Selecciona únicamente las variables
    utilizadas durante el entrenamiento.
    """

    return df[features + [target]].copy()


def split_features_target(
    df: pd.DataFrame,
    target: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa variables predictoras
    y variable objetivo.
    """

    X = df.drop(columns=target)

    y = df[target]

    return X, y


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Escala las variables numéricas.
    """

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns = X_train.columns,
        index = X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns = X_test.columns,
        index = X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler


def prepare_ml_dataset(
    filepath: str | Path
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepara el dataset
    para el entrenamiento.
    """

    dataset = load_dataset(filepath)

    dataset = select_features(
        dataset,
        FEATURES,
        TARGET
    )

    # Imputación de la primera variación porcentual
    dataset["Gas_Price_Pct_Change"] = (
    dataset["Gas_Price_Pct_Change"]
    .fillna(0))

    dataset = pd.get_dummies(
    dataset,
    columns=["Country"],
    drop_first=True
)

    X, y = split_features_target(
        dataset,
        TARGET
    )

    return X, y

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATA_PATH = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "dataset_final.csv"
    )

    X, y = prepare_ml_dataset(PROCESSED_DATA_PATH)


    print("DATASET PREPARADO PARA MACHINE LEARNING")
    print("-" * 60)

    print(f"Observaciones : {len(X)}")
    print(f"Variables     : {X.shape[1]}")
    print(f"Shape X       : {X.shape}")
    print(f"Shape y       : {y.shape}")

    print("\nVariables utilizadas:")
    print(X.columns.tolist())

    print("\nVariable objetivo:")
    print(TARGET)

    print("\nValores nulos:")
    print(X.isnull().sum())

    print(f"\nValores nulos en '{TARGET}': {y.isnull().sum()}")

    print("\nPrimeras observaciones:")
    print(X.head())



if __name__ == "__main__":
    main()

