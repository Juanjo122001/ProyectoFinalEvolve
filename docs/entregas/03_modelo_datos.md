# 03_modelo_datos.md

## 1. Resumen de la idea y datos del proyecto
El proyecto busca aportar luz sobre la volatilidad del mercado energético para ayudar a empresas a planificar su logística y presupuestos. A nivel de datos, la solución se divide en dos enfoques: primero, un análisis exploratorio apoyado en datos de Eurostat para demostrar la dependencia de la electricidad europea respecto al gas; y segundo, el desarrollo de un modelo de *Machine Learning* (Ridge / LightGBM) enfocado **exclusivamente en predecir la cotización diaria del gas natural** utilizando datos de MIBGAS.

## 2. Tecnología o formato de almacenamiento elegido
Se ha optado por trabajar con un sistema basado en **Ficheros CSV**.
*   **Justificación:** El volumen de datos esperado es manejable (datos tabulares, ~2000-5000 filas). Una base de datos relacional (PostgreSQL) o formatos distribuidos (Parquet) añadirían una complejidad de infraestructura innecesaria para el alcance del curso. Los CSV permiten una lectura inmediata con la librería `pandas` en Python, facilitando un prototipado rápido y compatibilidad absoluta en el repositorio.

## 3. Estructura de capas de datos
Los datos se organizarán bajo el estándar de arquitectura de datos analítica:

| Capa | Contenido esperado |
| :--- | :--- |
| `data/raw/` | Archivos CSV descargados directamente de Eurostat y MIBGAS, intocables e inmutables. |
| `data/interim/` | Archivos intermedios donde se han unificado formatos de fecha, traducido nombres de campos y filtrado anomalías. |
| `data/processed/` | Capa Gold. Datasets finales, listos para inyectar en los análisis y en los algoritmos predictivos (Scikit-Learn). |

## 4. Definición de la capa gold

Nuestra capa final consistirá en dos datasets principales tabulares, alojados en la carpeta `processed`:

| Dataset gold | Granularidad | Campos clave | Uso posterior |
| :--- | :--- | :--- | :--- |
| `gas_daily_prediction.csv` | Una fila por día de negociación. | `Trading_Day`, `Gas_Price`, `Gas_Price_Lag_1` | **Modelo predictivo temporal** (Ridge/LightGBM), *dashboard* y explicabilidad (SHAP). |
| `dataset_final.csv` | Una fila por semestre y país europeo. | `Year`, `Country`, `Electricity Price`, `Gas_Price` | **Análisis Exploratorio (EDA)** de la relación entre el gas y el mix eléctrico europeo. |

*Nota sobre `gas_daily_prediction.csv`:* Contiene el histórico estricto del mercado. Su clave primaria conceptual es la fecha (`Trading_Day`). El tipo de dato predominante es `float` (para precios y medias móviles) e `int` (para variables booleanas geopolíticas).

## 5. Relaciones entre datos
El proyecto consolida la información en tablas planas o sábanas analíticas (*flat tables*) en la capa final. 
*   **Justificación:** En *Machine Learning* aplicado a series temporales, los cruces de tablas dinámicas en el momento del entrenamiento aumentan el riesgo de *Data Leakage*. Por ello, los cruces (ej. unir el precio del gas histórico con el precio de la electricidad) se realizan programáticamente mediante `pandas.merge` durante la transición de la capa *Interim* a *Processed*, dejando los datasets finales completamente independientes y listos para su consumo.

## 6. Diccionario de datos inicial (Capa Gold - Gas)

| Campo | Descripción | Tipo de dato | Fuente | Obligatorio | Observaciones |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Trading_Day` | Fecha de la cotización | *Datetime* | MIBGAS | Sí | Índice temporal. |
| `Gas_Price` | Precio de cierre del gas | *Float* | MIBGAS | Sí | Variable Objetivo (Target). En €/MWh. |
| `Gas_Price_Lag_N` | Precio del gas en días pasados ($t-N$) | *Float* | Calculado | Sí | Variables inerciales principales. |
| `Gas_Rolling_Mean_N` | Media móvil de los últimos $N$ días | *Float* | Calculado | No | Suaviza el ruido intradía. |
| `Ukraine_War` | Conflicto activo (1 = Sí, 0 = No) | *Integer* | Externa | Sí | Variable *Dummy* geopolítica. |

## 7. Problemas de calidad esperados
Basado en la naturaleza de los mercados energéticos, se anticipan estos problemas reales:
*   **Valores nulos por festivos:** MIBGAS no opera fines de semana, dejando "huecos" que rompen la continuidad requerida por los modelos autorregresivos.
*   **Datos extremos (*Outliers*):** Picos masivos de volatilidad (ej. estallido de la guerra en marzo de 2022). No son errores de calidad, sino anomalías reales que los modelos deberán digerir sin sobreajustarse.
*   **Inconsistencia en tipos de datos:** Necesidad de convertir rigurosamente las fechas a objetos `datetime` de Pandas y manejar correctamente las codificaciones de los caracteres especiales europeos.

## 8. Decisiones de limpieza y transformación previstas
1.  **Tratamiento de nulos:** Se utilizará la técnica de interpolación o relleno hacia adelante (*Forward-Fill*) para los fines de semana, asumiendo que el precio del viernes se mantiene sábado y domingo.
2.  **Variables derivadas (*Feature Engineering*):** Creación de retardos temporales (*Lags* a 1, 7, 14 y 30 días), tendencias, estadísticos de ventana móvil (*Rolling*) y codificación cíclica (Seno/Coseno para los meses).
3.  **Codificación de categóricas:** *One-Hot Encoding* para países o estados en el dataset europeo.

## 9. Riesgos del modelo de datos
*   **Parte más clara:** El flujo de los datos diarios del mercado MIBGAS, ya que presenta una estructura rígida e inmutable ideal para series temporales.
*   **Parte con más incertidumbre:** Asegurar que, durante la generación recursiva de variables para la predicción a 30 días, no se filtre información del futuro hacia los *lags* o las medias móviles.
*   **Fuente más problemática:** La unificación del dataset europeo (Eurostat) debido a las diferentes bandas de consumo y regímenes fiscales de cada país.