import pandas as pd
from pathlib import Path
from preprocessing import prepare_ml_dataset

def temporal_train_test_split(
        X: pd.DataFrame,
        y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el dataset en conjunto de entrenamiento y prueba
    respetando el orden temporal.
    """

    last_year = X["Year"].max()
    train_mask = X["Year"] < last_year
    test_mask = X["Year"] == last_year

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()

    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()

    return X_train, X_test, y_train, y_test

def main():

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    processed_data_path = (PROJECT_ROOT / "data" / "processed" / "dataset_final.csv")

    X, y = prepare_ml_dataset(processed_data_path)

    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y)

    print("DIVISIÓN TEMPORAL DEL DATASET")
    print('-' * 60)

    
    print(f"Entrenamiento: {len(X_train)} observaciones")
    print(f"Test: {len(X_test)} observaciones")

    print(f"\nShape X_train: {X_train.shape}")
    print(f"Shape X_test: {X_test.shape}")

    print(f"\nShape y_train: {y_train.shape}")
    print(f"Shape y_test: {y_test.shape}")

    print("\nAños entrenamiento:")
    print(sorted(X_train["Year"].unique()))

    print("\nAños prueba:")
    print(sorted(X_test["Year"].unique()))


if __name__ == "__main__":
    main()


