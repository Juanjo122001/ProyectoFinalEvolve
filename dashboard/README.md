# VOLT · Energy Intelligence — Dashboard del TFM

Dashboard interactivo (Dash + Plotly) que presenta el TFM *«Análisis y predicción de precios en los
mercados energéticos de electricidad y gas natural»*. Lee directamente los datos y modelos que genera
el pipeline del proyecto (`data/processed/` y `results/`); no necesita ficheros adicionales.

## Cómo ejecutarlo

Opción rápida (Windows): doble clic en `dashboard/run_dashboard.bat`.

Manual, desde la raíz del proyecto:

```bash
.venv\Scripts\activate          # entorno del TFM
pip install -r dashboard/requirements.txt
python dashboard/app.py
```

Abrir <http://127.0.0.1:8050>. El arranque tarda unos segundos: precalcula la predicción a 30 días,
el backtesting y los valores SHAP.

## Secciones

| Página | Contenido |
|---|---|
| **Resumen** | KPIs, serie diaria MIBGAS con los periodos de crisis, 5 hallazgos y cronología geopolítica |
| **Mercado europeo** | España vs media UE, España/Portugal vs gas, mapa y ranking de 40 países, dispersión y distribución. Filtro de impuestos y selector de países |
| **Gas ↔ Electricidad** | Matriz de Spearman (Ilustración 19), dispersión semestral y Tablas 1–2 |
| **Modelos de electricidad** | Tabla 6, R² por modelo e importancia de variables del Random Forest |
| **Predicción del gas** | Proyección recursiva a 30 días (LightGBM, Ridge, Naïve), backtesting (Tabla 7), hold-out y simulador de coste |
| **Explicabilidad** | SHAP global, beeswarm y descomposición de la predicción de mañana |
| **Metodología** | Pipeline, datasets, variables, limitaciones y trabajo futuro |

## Detalles técnicos

- **Modelos**: carga `results/models/gas_lightgbm.pkl` y `gas_ridge.pkl`. Si no se pudieran cargar
  (p. ej. otra versión de scikit-learn) los reentrena con los mismos hiperparámetros del TFM.
- **Predicción y backtesting**: réplica exacta de `src/models/predict_gas_future.py` y
  `src/models/backtesting.py` (5 orígenes en el 20 % final, horizontes 1–30).
- **SHAP sin dependencias extra**: TreeSHAP exacto nativo de LightGBM (`pred_contrib=True`) y forma
  cerrada lineal para Ridge (`coef · (x − μ) / σ`).
- **Tema claro/oscuro**: botón en la barra lateral (el claro se ve mejor en proyector).
- **Mapa sin conexión**: la geometría de Europa va en `assets/topojson/`.
- Cualquier gráfico se puede exportar a PNG en alta resolución desde el icono de cámara (al pasar el ratón).

## Estructura

```
dashboard/
├── app.py        # layout, páginas y callbacks
├── data.py       # carga de datos, modelos, predicción, backtesting y SHAP
├── charts.py     # figuras Plotly
├── theme.py      # tokens de color (oscuro / claro)
├── assets/       # CSS y topojson
└── requirements.txt
```
