"""
Motor de datos del dashboard VOLT.

Lee los datasets y modelos generados por el pipeline del TFM
(data/processed, results/) y calcula todo lo que muestra el dashboard:
agregados de Eurostat, correlaciones, predicción recursiva a 30 días,
backtesting y valores SHAP.

Todo se calcula una sola vez al arrancar (lru_cache) para que la
navegación sea instantánea.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Rutas del proyecto
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"

GAS_TARGET = "Gas_Price"
FORECAST_DAYS = 30
TEST_SIZE = 0.20

# ------------------------------------------------------------------
# Acontecimientos geopolíticos (mismas fechas que el pipeline)
# ------------------------------------------------------------------
EVENTS = [
    {"key": "crisis", "label": "Crisis energética", "start": "2021-09-01", "end": "2023-03-31",
     "text": "Recuperación post-COVID, reservas de gas en mínimos históricos."},
    {"key": "ukraine", "label": "Invasión de Ucrania", "start": "2022-02-24", "end": None,
     "text": "Rusia suministraba ~1/4 de la energía consumida en Europa (2021)."},
    {"key": "iberian", "label": "Excepción Ibérica", "start": "2022-06-15", "end": "2023-12-31",
     "text": "Tope al precio del gas para generación eléctrica en España y Portugal."},
    {"key": "mideast", "label": "Conflicto Oriente Medio", "start": "2023-10-07", "end": None,
     "text": "Daños en infraestructura de GNL y tráfico limitado en Ormuz."},
]

TAX_OPTIONS = {
    "all": "Todas las modalidades (memoria)",
    "Excluding taxes and levies": "Sin impuestos",
    "Excluding VAT and other recoverable taxes and levies": "Sin IVA",
    "All taxes and levies included": "Con todos los impuestos",
}

COUNTRY_ES = {
    "Albania": "Albania", "Austria": "Austria", "Belgium": "Bélgica",
    "Bosnia and Herzegovina": "Bosnia y Herz.", "Bulgaria": "Bulgaria",
    "Croatia": "Croacia", "Cyprus": "Chipre", "Czechia": "Chequia",
    "Denmark": "Dinamarca", "Estonia": "Estonia", "Finland": "Finlandia",
    "France": "Francia", "Georgia": "Georgia", "Germany": "Alemania",
    "Greece": "Grecia", "Hungary": "Hungría", "Iceland": "Islandia",
    "Ireland": "Irlanda", "Italy": "Italia", "Kosovo*": "Kosovo",
    "Latvia": "Letonia", "Liechtenstein": "Liechtenstein",
    "Lithuania": "Lituania", "Luxembourg": "Luxemburgo", "Malta": "Malta",
    "Moldova": "Moldavia", "Montenegro": "Montenegro",
    "Netherlands": "Países Bajos", "North Macedonia": "Macedonia del N.",
    "Norway": "Noruega", "Poland": "Polonia", "Portugal": "Portugal",
    "Romania": "Rumanía", "Serbia": "Serbia", "Slovakia": "Eslovaquia",
    "Slovenia": "Eslovenia", "Spain": "España", "Sweden": "Suecia",
    "Türkiye": "Turquía", "Ukraine": "Ucrania",
}

COUNTRY_ISO3 = {
    "Albania": "ALB", "Austria": "AUT", "Belgium": "BEL",
    "Bosnia and Herzegovina": "BIH", "Bulgaria": "BGR", "Croatia": "HRV",
    "Cyprus": "CYP", "Czechia": "CZE", "Denmark": "DNK", "Estonia": "EST",
    "Finland": "FIN", "France": "FRA", "Georgia": "GEO", "Germany": "DEU",
    "Greece": "GRC", "Hungary": "HUN", "Iceland": "ISL", "Ireland": "IRL",
    "Italy": "ITA", "Kosovo*": "XKX", "Latvia": "LVA",
    "Liechtenstein": "LIE", "Lithuania": "LTU", "Luxembourg": "LUX",
    "Malta": "MLT", "Moldova": "MDA", "Montenegro": "MNE",
    "Netherlands": "NLD", "North Macedonia": "MKD", "Norway": "NOR",
    "Poland": "POL", "Portugal": "PRT", "Romania": "ROU", "Serbia": "SRB",
    "Slovakia": "SVK", "Slovenia": "SVN", "Spain": "ESP", "Sweden": "SWE",
    "Türkiye": "TUR", "Ukraine": "UKR",
}

FEATURE_ES = {
    "Gas_Price_Lag_1": "Precio gas t-1",
    "Gas_Price_Lag_7": "Precio gas t-7",
    "Gas_Price_Lag_14": "Precio gas t-14",
    "Gas_Price_Lag_30": "Precio gas t-30",
    "Gas_Rolling_Mean_7": "Media móvil 7d",
    "Gas_Rolling_Mean_30": "Media móvil 30d",
    "Gas_Rolling_Std_7": "Volatilidad 7d",
    "Gas_Trend": "Tendencia (t-1 − t-7)",
    "Gas_Price_Pct_Change_Lag_1": "Variación % t-1",
    "Year": "Año", "Month": "Mes", "Day": "Día", "Day_of_Week": "Día semana",
    "Day_of_Year": "Día del año", "Quarter": "Trimestre",
    "Semester_Number": "Semestre", "Month_Sin": "Mes (sin)",
    "Month_Cos": "Mes (cos)", "Day_of_Week_Sin": "Día semana (sin)",
    "Day_of_Week_Cos": "Día semana (cos)", "Ukraine_War": "Guerra Ucrania",
    "Energy_Crisis": "Crisis energética",
    "Middle_East_Conflict": "Conflicto O. Medio",
    "Gas_Price": "Precio gas", "Gas_Price_Pct_Change": "Variación % gas",
    "Electricity Price": "Precio electricidad",
}


def feature_label(name: str) -> str:
    if name.startswith("Country_"):
        c = name.replace("Country_", "")
        return f"País: {COUNTRY_ES.get(c, c)}"
    return FEATURE_ES.get(name, name)


# ==================================================================
# 1. EUROSTAT + MIBGAS (semestral)
# ==================================================================
@lru_cache(maxsize=1)
def load_europe() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "dataset_final.csv", sep=";")
    df = df.rename(columns={"Electricity Price": "Price"})
    df["Country_ES"] = df["Country"].map(COUNTRY_ES).fillna(df["Country"])
    df["ISO3"] = df["Country"].map(COUNTRY_ISO3)
    df["SemKey"] = df["Year"] * 10 + df["Semester_Number"]
    return df


def semesters() -> list[str]:
    df = load_europe()
    return df.sort_values("SemKey")["Semester"].drop_duplicates().tolist()


def filter_tax(tax: str) -> pd.DataFrame:
    df = load_europe()
    if tax and tax != "all":
        df = df[df["tax_type"] == tax]
    return df


def gas_by_semester() -> pd.DataFrame:
    df = load_europe()
    return (df.groupby(["SemKey", "Semester"], as_index=False)
              .agg(Gas=("Gas_Price", "first"), Gas_Max=("Gas_Price_Max", "first"),
                   Gas_Min=("Gas_Price_Min", "first"))
              .sort_values("SemKey"))


def europe_mean_by_semester(tax: str) -> pd.DataFrame:
    df = filter_tax(tax)
    return (df.groupby(["SemKey", "Semester"], as_index=False)["Price"].mean()
              .sort_values("SemKey"))


def country_by_semester(tax: str, countries: list[str]) -> pd.DataFrame:
    df = filter_tax(tax)
    df = df[df["Country"].isin(countries)]
    return (df.groupby(["Country", "Country_ES", "SemKey", "Semester"], as_index=False)["Price"]
              .mean().sort_values(["Country", "SemKey"]))


def country_ranking(tax: str, semester: str | None = None) -> pd.DataFrame:
    df = filter_tax(tax)
    if semester and semester != "all":
        df = df[df["Semester"] == semester]
    out = (df.groupby(["Country", "Country_ES", "ISO3"], as_index=False)["Price"].mean()
             .sort_values("Price", ascending=False).reset_index(drop=True))
    out["Rank"] = np.arange(1, len(out) + 1)
    return out


def country_prices(tax: str, countries: list[str]) -> pd.DataFrame:
    df = filter_tax(tax)
    return df[df["Country"].isin(countries)][["Country", "Country_ES", "Semester", "Price"]]


def iberia_semester(tax: str) -> pd.DataFrame:
    df = filter_tax(tax)
    df = df[df["Country"].isin(["Spain", "Portugal"])]
    return (df.groupby(["SemKey", "Semester"], as_index=False)
              .agg({"Price": "mean", "Gas_Price": "first", "Gas_Trend": "first",
                    "Gas_Price_Pct_Change": "first", "Ukraine_War": "first",
                    "Energy_Crisis": "first", "Middle_East_Conflict": "first"})
              .sort_values("SemKey"))


def spearman_iberia(tax: str) -> pd.DataFrame:
    """Réplica de la matriz de Spearman de la memoria (Ilustración 19)."""
    cols = ["Price", "Gas_Price", "Gas_Trend", "Gas_Price_Pct_Change",
            "Ukraine_War", "Energy_Crisis", "Middle_East_Conflict"]
    corr = iberia_semester(tax)[cols].corr(method="spearman")
    labels = {"Price": "Electricidad", "Gas_Price": "Precio gas", "Gas_Trend": "Tendencia gas",
              "Gas_Price_Pct_Change": "Var. % gas", "Ukraine_War": "Guerra Ucrania",
              "Energy_Crisis": "Crisis energética", "Middle_East_Conflict": "Conflicto O. Medio"}
    return corr.rename(index=labels, columns=labels)


@lru_cache(maxsize=1)
def semester_summary() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "semester_summary.csv", sep=";")
    return df.sort_values(["Year", "Semester_Number"])


# ==================================================================
# 2. MODELOS DE ELECTRICIDAD (resultados guardados por el pipeline)
# ==================================================================
@lru_cache(maxsize=1)
def electricity_model_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "model_results.csv")


@lru_cache(maxsize=1)
def electricity_feature_importance() -> pd.DataFrame:
    fi = pd.read_csv(RESULTS_DIR / "feature_importance.csv", sep=";")
    fi["Label"] = fi["Feature"].map(feature_label)
    fi["Group"] = np.where(fi["Feature"].str.startswith("Country_"), "País",
                  np.where(fi["Feature"].str.startswith("Gas"), "Mercado del gas",
                  np.where(fi["Feature"].isin(["Ukraine_War", "Energy_Crisis", "Middle_East_Conflict"]),
                           "Geopolítica", "Temporal")))
    return fi.sort_values("Importance", ascending=False)


# ==================================================================
# 3. GAS DIARIO (MIBGAS) — modelos, predicción, backtesting, SHAP
# ==================================================================
@lru_cache(maxsize=1)
def load_gas() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "gas_daily_prediction.csv", sep=";")
    df["Trading_Day"] = pd.to_datetime(df["Trading_Day"])
    return df.sort_values("Trading_Day").reset_index(drop=True)


def gas_feature_columns() -> list[str]:
    excluded = {"Trading_Day", GAS_TARGET, "Gas_Price_Pct_Change"}
    return [c for c in load_gas().columns if c not in excluded]


def split_index() -> int:
    return int(len(load_gas()) * (1 - TEST_SIZE))


def _expected_features(model) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return gas_feature_columns()


def _train_fallback(name: str):
    """Reentrena con los mismos hiperparámetros del TFM si el .pkl no carga."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = load_gas()
    feats = gas_feature_columns()
    X, y = df[feats].iloc[:split_index()], df[GAS_TARGET].iloc[:split_index()]
    if name == "Ridge Regression":
        model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    else:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              n_jobs=-1, verbosity=-1)
    model.fit(X, y)
    return model


@lru_cache(maxsize=1)
def gas_models() -> dict:
    files = {"LightGBM": "gas_lightgbm.pkl", "Ridge Regression": "gas_ridge.pkl"}
    models = {}
    for name, fname in files.items():
        source = "pkl"
        try:
            model = joblib.load(MODELS_DIR / fname)
        except Exception:  # versión distinta o fichero ausente
            model, source = _train_fallback(name), "reentrenado"
        models[name] = {"model": model, "features": _expected_features(model), "source": source}
    return models


@lru_cache(maxsize=1)
def gas_holdout_metrics() -> pd.DataFrame:
    path = RESULTS_DIR / "metrics" / "gas_model_results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def gas_test_predictions() -> pd.DataFrame:
    """Predicción a un paso (t+1) sobre el 20 % final de la serie."""
    df = load_gas()
    test = df.iloc[split_index():].copy()
    out = test[["Trading_Day", GAS_TARGET]].rename(columns={GAS_TARGET: "Real"})
    for name, info in gas_models().items():
        out[name] = info["model"].predict(test[info["features"]])
    out["Naive"] = df[GAS_TARGET].shift(1).iloc[split_index():].values
    return out.reset_index(drop=True)


def _dynamic_features(history: list[float]) -> dict:
    return {
        "Gas_Price_Lag_1": history[-1],
        "Gas_Price_Lag_7": history[-7],
        "Gas_Price_Lag_14": history[-14],
        "Gas_Price_Lag_30": history[-30],
        "Gas_Rolling_Mean_7": float(np.mean(history[-7:])),
        "Gas_Rolling_Mean_30": float(np.mean(history[-30:])),
        "Gas_Rolling_Std_7": float(np.std(history[-7:], ddof=1)),
        "Gas_Trend": history[-1] - history[-7],
        "Gas_Price_Pct_Change_Lag_1": ((history[-1] / history[-2]) - 1) * 100 if history[-2] else 0.0,
    }


def _calendar_features(date: pd.Timestamp) -> dict:
    return {
        "Year": date.year, "Month": date.month, "Day": date.day,
        "Day_of_Week": date.dayofweek, "Day_of_Year": date.dayofyear,
        "Quarter": date.quarter, "Semester_Number": (date.month - 1) // 6 + 1,
        "Month_Sin": np.sin(2 * np.pi * date.month / 12),
        "Month_Cos": np.cos(2 * np.pi * date.month / 12),
        "Day_of_Week_Sin": np.sin(2 * np.pi * date.dayofweek / 7),
        "Day_of_Week_Cos": np.cos(2 * np.pi * date.dayofweek / 7),
        "Ukraine_War": int(date >= pd.Timestamp("2022-02-24")),
        "Energy_Crisis": int(pd.Timestamp("2021-09-01") <= date <= pd.Timestamp("2023-03-31")),
        "Middle_East_Conflict": int(date >= pd.Timestamp("2023-10-07")),
    }


@lru_cache(maxsize=1)
def future_forecast() -> pd.DataFrame:
    """Predicción recursiva a 30 días (misma lógica que predict_gas_future.py)."""
    df = load_gas()
    last_date = df["Trading_Day"].iloc[-1]
    dates = [last_date + pd.Timedelta(days=i) for i in range(1, FORECAST_DAYS + 1)]
    out = pd.DataFrame({"Trading_Day": dates})
    rows_first = {}
    for name, info in gas_models().items():
        history = df[GAS_TARGET].tolist()
        preds = []
        for i, d in enumerate(dates):
            row = {**_calendar_features(d), **_dynamic_features(history)}
            X = pd.DataFrame([row])[info["features"]]
            if i == 0:
                rows_first[name] = X
            p = float(info["model"].predict(X)[0])
            preds.append(p)
            history.append(p)
        out[name] = preds
    out["Naive"] = float(df[GAS_TARGET].iloc[-1])
    future_forecast.first_rows = rows_first  # para explicar t+1
    return out


@lru_cache(maxsize=1)
def backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting recursivo a 30 días desde 5 orígenes del conjunto de test
    (réplica de src/models/backtesting.py). Devuelve:
      - métricas por horizonte (1..30) y modelo
      - trayectorias de cada origen para visualizarlas
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    df = load_gas()
    start = len(df) - int(len(df) * TEST_SIZE)  # idéntico a backtesting.py
    origins = np.linspace(start, len(df) - 31, 5, dtype=int)
    records, paths = [], []
    for o in origins:
        actual = df.loc[o:o + 29, GAS_TARGET].values
        naive = np.full(30, df.loc[o - 1, GAS_TARGET])
        preds_all = {"Naive": naive}
        for name, info in gas_models().items():
            history = df.loc[:o - 1, GAS_TARGET].tolist()
            preds = []
            for step in range(30):
                X = df.iloc[[o + step]].copy()
                for k, v in _dynamic_features(history).items():
                    X[k] = v
                p = float(info["model"].predict(X[info["features"]])[0])
                preds.append(p)
                history.append(p)
            preds_all[name] = np.array(preds)
        for name, preds in preds_all.items():
            for h in range(30):
                records.append({"Origin": df.loc[o, "Trading_Day"], "Model": name, "Horizon": h + 1,
                                "Actual": actual[h], "Predicted": preds[h]})
        paths.append({"origin": df.loc[o, "Trading_Day"],
                      "dates": df.loc[o:o + 29, "Trading_Day"].tolist(),
                      "actual": actual, **{k: v for k, v in preds_all.items()}})
    res = pd.DataFrame(records)
    rows = []
    for (m, h), g in res.groupby(["Model", "Horizon"]):
        mse = mean_squared_error(g["Actual"], g["Predicted"])
        rows.append({"Model": m, "Horizon": h, "MAE": mean_absolute_error(g["Actual"], g["Predicted"]),
                     "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(g["Actual"], g["Predicted"])})
    return pd.DataFrame(rows), paths


# -------------------------- SHAP ----------------------------------
@lru_cache(maxsize=2)
def shap_values(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Valores SHAP sobre el conjunto de test.
      - LightGBM: TreeSHAP exacto nativo (pred_contrib=True).
      - Ridge: SHAP lineal = coef · (x − μ_train) / σ_train.
    Devuelve (shap, X, base_value).
    """
    df = load_gas()
    info = gas_models()[model_name]
    X = df[info["features"]].iloc[split_index():].reset_index(drop=True)
    return _shap_for(model_name, X)


def _shap_for(model_name: str, X: pd.DataFrame):
    info = gas_models()[model_name]
    model = info["model"]
    if model_name == "LightGBM":
        contrib = model.predict(X, pred_contrib=True)
        sv = pd.DataFrame(contrib[:, :-1], columns=X.columns)
        base = float(contrib[0, -1])
    else:
        scaler = model.named_steps["scaler"]
        reg = model.named_steps["model"]
        z = (X.values - scaler.mean_) / scaler.scale_
        sv = pd.DataFrame(z * reg.coef_, columns=X.columns)
        base = float(reg.intercept_)
    return sv, X, base


def shap_importance(model_name: str) -> pd.DataFrame:
    sv, _, _ = shap_values(model_name)
    imp = sv.abs().mean().sort_values(ascending=False)
    out = imp.reset_index()
    out.columns = ["Feature", "MeanAbsSHAP"]
    out["Share"] = out["MeanAbsSHAP"] / out["MeanAbsSHAP"].sum()
    out["Label"] = out["Feature"].map(feature_label)
    return out


def explain_next_day(model_name: str) -> tuple[pd.DataFrame, float, float]:
    """Contribuciones SHAP de la predicción para t+1 (primer día futuro)."""
    future_forecast()
    X = future_forecast.first_rows[model_name]
    sv, _, base = _shap_for(model_name, X)
    s = sv.iloc[0]
    out = pd.DataFrame({"Feature": s.index, "SHAP": s.values, "Value": X.iloc[0].values})
    out["Label"] = out["Feature"].map(feature_label)
    out = out.reindex(out["SHAP"].abs().sort_values(ascending=False).index)
    pred = base + s.sum()
    return out, base, pred


# ==================================================================
# 4. KPIs globales
# ==================================================================
@lru_cache(maxsize=1)
def headline() -> dict:
    gas = load_gas()
    last = gas.iloc[-1]
    prev_week = gas.iloc[-8]
    peak = gas.loc[gas[GAS_TARGET].idxmax()]
    rf = electricity_model_results().iloc[0]
    corr = spearman_iberia("all").loc["Electricidad", "Precio gas"]
    bt, _ = backtest()
    ff = future_forecast()
    return {
        "last_price": float(last[GAS_TARGET]),
        "last_date": last["Trading_Day"],
        "week_change": float(last[GAS_TARGET] / prev_week[GAS_TARGET] - 1),
        "peak_price": float(peak[GAS_TARGET]),
        "peak_date": peak["Trading_Day"],
        "rf_r2": float(rf["R2"]), "rf_mae": float(rf["MAE"]),
        "spearman": float(corr),
        "n_days": len(gas), "n_obs_eu": len(load_europe()),
        "n_countries": load_europe()["Country"].nunique(),
        "gas_start": gas["Trading_Day"].iloc[0], "gas_end": last["Trading_Day"],
        "forecast_end": ff["Trading_Day"].iloc[-1],
    }


def warmup() -> None:
    """Precalcula todo al arrancar."""
    load_europe(); load_gas(); gas_models(); gas_test_predictions()
    future_forecast(); backtest(); headline()
    for m in ("LightGBM", "Ridge Regression"):
        shap_values(m)
