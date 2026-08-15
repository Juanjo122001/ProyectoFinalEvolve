import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from preprocessing import prepare_ml_dataset, scale_features
from dataset_split import temporal_train_test_split
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def save_model(
    model,
    model_name: str,
    project_root: Path
):
    models_path = (
        project_root
        / "results"
        / "models"
    )

    models_path.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        models_path / f"{model_name}.pkl"
    )

    print(f"Modelo guardado: {model_name}")

def evaluate_model(
    model,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict[str, float]:
    """
    Evalúa el modelo en el conjunto de prueba.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Modelo entrenado.
    X_test : pd.DataFrame
        Conjunto de prueba de variables predictoras.
    y_test : pd.Series
        Conjunto de prueba de variable objetivo.

    Returns
    -------
    pd.DataFrame
        Métricas de evaluación del modelo.
    """

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results = pd.DataFrame({
        "Model": [model_name],
        "MSE": [mse],
        "RMSE": [rmse],
        "MAE": [mae],
        "R2": [r2]
    })

    return results

def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> LinearRegression:
    """
    Entrena un modelo de regresión lineal.

    Parameters
    ----------
    X_train : pd.DataFrame
        Conjunto de entrenamiento de variables predictoras.
    y_train : pd.Series
        Conjunto de entrenamiento de variable objetivo.

    Returns
    -------
    LinearRegression
        Modelo entrenado.
    """

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def train_best_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Ridge:
    """
    Selecciona el mejor alpha para Ridge Regression
    y además entrena el modelo con ese alpha

    Parameters
    ----------
    X_train : pd.DataFrame
        Conjunto de entrenamiento de variables predictoras.
    y_train : pd.Series
        Conjunto de entrenamiento de variable objetivo.

    Returns
    -------
    Ridge
        Modelo entrenado.
    """

    alphas = [0.01, 0.1, 1, 10, 100, 1000]
    best_alpha = None
    best_model = None
    best_score = float('-inf')

    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_train, y_train)

        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_model = model

    print(f"\nMejor alpha para Ridge Regression: {best_alpha}")
    print(f"R2 score en el conjunto de entrenamiento: {best_score:.4f}")


    return best_model

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestRegressor:
    """
    Entrena un modelo de Random Forest.

    Parameters
    ----------
    X_train : pd.DataFrame
        Conjunto de entrenamiento de variables predictoras.
    y_train : pd.Series
        Conjunto de entrenamiento de variable objetivo.
    Returns
    -------
    RandomForestRegressor
        Modelo entrenado.
    """

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model

def optimize_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestRegressor:
    """
    Optimiza los hiperparámetros de un modelo de Random Forest utilizando GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Conjunto de entrenamiento de variables predictoras.
    y_train : pd.Series
        Conjunto de entrenamiento de variable objetivo.

    Returns
    -------
    RandomForestRegressor
        Modelo entrenado con los mejores hiperparámetros.
    """

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='r2',
        cv=TimeSeriesSplit(n_splits=5),
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    print(f"\nMejores hiperparámetros para Random Forest: {grid_search.best_params_}")
    print()
    print("Mejor R2 CV SCORE:")
    print(grid_search.best_score_)
    
    return grid_search.best_estimator_

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> XGBRegressor:
    """
    Entrena un modelo de XGBoost.

    Parameters
    ----------
    X_train : pd.DataFrame
        Conjunto de entrenamiento de variables predictoras.
    y_train : pd.Series
        Conjunto de entrenamiento de variable objetivo.

    Returns
    -------
    XGBRegressor
        Modelo entrenado.
    """

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> LGBMRegressor:
    """
    Entrena un modelo de LightGBM.

    Parameters
    ----------
    X_train : pd.DataFrame
        Conjunto de entrenamiento de variables predictoras.
    y_train : pd.Series
        Conjunto de entrenamiento de variable objetivo.

    Returns
    -------
    LGBMRegressor
        Modelo entrenado.
    """

    model = LGBMRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='regression',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model

def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> CatBoostRegressor:
    """
    Entrena un modelo de CatBoost.

    Parameters
    ----------
    X_train : pd.DataFrame
        Conjunto de entrenamiento de variables predictoras.
    y_train : pd.Series
        Conjunto de entrenamiento de variable objetivo.

    Returns
    -------
    CatBoostRegressor
        Modelo entrenado.
    """

    model = CatBoostRegressor(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        loss_function='RMSE',
        random_seed=42,
        verbose=False
    )
    model.fit(X_train, y_train)

    return model

def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> MLPRegressor:
    """
    Entrena un modelo de MLP (Multi-Layer Perceptron).

    Parameters
    ----------
    X_train : pd.DataFrame
        Conjunto de entrenamiento de variables predictoras.
    y_train : pd.Series
        Conjunto de entrenamiento de variable objetivo.

    Returns
    -------
    MLPRegressor
        Modelo entrenado.
    """

    model = MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation='relu',
        solver='adam',
        alpha = 0.001,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    processed_data_path = (PROJECT_ROOT / "data" / "processed" / "dataset_final.csv")

    # Cargar y preparar el dataset
    X, y = prepare_ml_dataset(processed_data_path)

    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)


    # Entrenamiento y evaluación de modelos
    results = pd.DataFrame()
    print(X_train.head())  # Mostrar las primeras filas de X_train para verificar la carga de datos

    
    # Linear Regression

    lr = train_linear_regression(
        X_train_scaled,
        y_train
    )

    results = pd.concat(
        [
            results,
            evaluate_model(
                lr,
                "Linear Regression",
                X_test_scaled,
                y_test
            )
        ],
        ignore_index=True
    )

    # Ridge Regression
    rr = train_best_ridge(
        X_train_scaled,
        y_train
    )

    results = pd.concat(
        [
            results,
            evaluate_model(
                rr,
                "Ridge Regression",
                X_test_scaled,
                y_test
            )
        ],
        ignore_index=True
    )

   
    # Random Forest

    rf = train_random_forest(
        X_train,
        y_train
    )

    results = pd.concat(
        [
            results,
            evaluate_model(
                rf,
                "Random Forest",
                X_test,
                y_test
            )
        ],
        ignore_index=True
    )

    # Grid Search Random Forest
    
    optimized_rf = optimize_random_forest(
        X_train,
        y_train
    )

    results = pd.concat(
        [
            results,
            evaluate_model(
                optimized_rf,
                "Optimized Random Forest",
                X_test,
                y_test
            )
        ],
        ignore_index=True
    )


    # XGBoost

    xgb = train_xgboost(
        X_train,
        y_train
    )

    results = pd.concat(
        [
            results,
            evaluate_model(
                xgb,
                "XGBoost",
                X_test,
                y_test
            )
        ],
        ignore_index=True
    )


    # LightGBM

    lgbm = train_lightgbm(
        X_train,
        y_train
    )

    results = pd.concat(
        [
            results,
            evaluate_model(
                lgbm,
                "LightGBM",
                X_test,
                y_test
            )
        ],
        ignore_index=True
    )


    # CatBoost

    cat = train_catboost(
        X_train,
        y_train
    )

    results = pd.concat(
        [
            results,
            evaluate_model(
                cat,
                "CatBoost",
                X_test,
                y_test
            )
        ],
        ignore_index=True
    )


    # MLP

    mlp = train_mlp(
        X_train_scaled,
        y_train
    )

    results = pd.concat(
        [
            results,
            evaluate_model(
                mlp,
                "MLP",
                X_test_scaled,
                y_test
            )
        ],
        ignore_index=True
    )


    # Resultados finales

    results = (
        results
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    print("\n")
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("-" * 60)
    print(results)

    # =====================================================
    # Guardar resultados
    # =====================================================

    results_path = (
        PROJECT_ROOT
        / "results"
    )

    results_path.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        results_path / "model_results.csv",
        index=False
    )

    print("\nResultados guardados en:")
    print(results_path / "model_results.csv")

    trained_models = {

    "linear_regression": lr,

    "ridge_regression": rr,

    "random_forest": rf,

    "optimized_random_forest": optimized_rf,

    "xgboost": xgb,

    "lightgbm": lgbm,

    "catboost": cat,

    "mlp": mlp

}
    for model_name, model in trained_models.items():

     save_model(

         model,

         model_name,

         PROJECT_ROOT

     )


if __name__ == "__main__":
    main()