# 03_modelo_datos.md

## 1. Resumen de la idea y datos del proyecto
El proyecto busca predecir el precio del mercado energético (electricidad y gas) para ayudar a empresas a anticipar sobrecostes y planificar su logística y presupuestos. La solución técnica será un modelo de *Machine Learning* (Random Forest / LightGBM) que analice series temporales. Las fuentes principales serán Eurostat (precios eléctricos europeos) y MIBGAS (cotización diaria del gas natural), aportando datos macroeconómicos de alta fiabilidad.

## 2. Tecnología o formato de almacenamiento elegido
Se ha optado por trabajar con un sistema basado en **Ficheros CSV**.
*   **Justificación:** El volumen de datos esperado es de unos pocos megabytes (datos tabulares, ~2000-5000 filas). Para este tamaño, una base de datos relacional (PostgreSQL) o formatos distribuidos (Parquet) añadirían una complejidad de infraestructura innecesaria para el alcance del curso. Los CSV permiten una lectura inmediata con la librería `pandas` en Python, facilitando un prototipado rápido, versiones controlables en Git y compatibilidad absoluta.

## 3. Estructura de capas de datos
Los datos se organizarán bajo el estándar de arquitectura de datos analítica:

| Capa | Contenido esperado |
| :--- | :--- |
| `data/raw/` | Archivos CSV y Excel descargados directamente de Eurostat y MIBGAS, intocables e inmutables. |
| `data/processed/` | Archivos intermedios donde se han unificado formatos de fecha, eliminado columnas irrelevantes, imputado nulos de fines de semana y traducido nombres de campos. |
| `data/gold/` | Datasets finales, enriquecidos con *Feature Engineering* (creación de medias móviles, variables de retardo/lags y *dummies* geopolíticas). Listos para inyectar en Scikit-Learn. |

## 4. Definición de la capa gold

Nuestra capa final consistirá en dos datasets principales tabulares (uno para el análisis eléctrico y otro para la predicción del gas). A continuación se detalla el dataset principal de predicción:

| Dataset gold | Granularidad | Campos clave | Uso posterior |
| :--- | :--- | :--- | :--- |
| `gas_daily_prediction.csv` | Una fila por día de negociación en el mercado. | `Trading_Day`, `Gas_Price`, `Gas_Price_Lag_1` | Modelo predictivo temporal (LightGBM/Ridge), *dashboard* analítico y SHAP. |
| `dataset_final.csv` | Una fila por semestre y país europeo. | `Year`, `Country`, `Electricity_Price`, `Gas_Price` | EDA geográfico, modelo de regresión espacial (Random Forest). |

*Nota sobre `gas_daily_prediction.csv`:* Contendrá aprox. 1500-2000 registros. Su clave primaria conceptual es la fecha (`Trading_Day`). El tipo de dato predominante será `float` (para precios y medias) e `int` (para variables binarias/indicadores).

## 5. Relaciones entre datos
El proyecto consolida la información en tablas planas o sábana analíticas (*flat tables*) en la capa Gold. 
*   **Justificación de único dataset por modelo:** En *Machine Learning* aplicado a series temporales, las bases de datos relacionales en el momento del entrenamiento son contraproducentes y aumentan el riesgo de *Data Leakage*. Por ello, los cruces (ej. unir el precio del gas histórico con el precio de la electricidad de un país concreto) se realizarán programáticamente durante el paso de la capa *Processed* a la *Gold* mediante `LEFT JOIN` utilizando las fechas (`Trading_Day`) o identificadores espaciales (`Country`) como claves.

## 6. Diccionario de datos inicial (Capa Gold - Gas)

| Campo | Descripción | Tipo de dato | Fuente | Obligatorio | Observaciones |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Trading_Day` | Fecha de la cotización | *Date* | MIBGAS | Sí | Formato AAAA-MM-DD. |
| `Gas_Price` | Precio de cierre del gas | *Float* | MIBGAS | Sí | Variable Objetivo (Target). En €/MWh. |
| `Gas_Price_Lag_1` | Precio del gas el día anterior ($t-1$) | *Float* | Calculado | Sí | Variable inercial principal. |
| `Gas_Rolling_Mean_7` | Media móvil de los últimos 7 días | *Float* | Calculado | No | Suaviza el ruido intradía. |
| `Ukraine_War` | Conflicto activo (1 = Sí, 0 = No) | *Integer* | Externa | Sí | Variable *Dummy* geopolítica. |

## 7. Problemas de calidad esperados
Basado en la naturaleza de los mercados energéticos, se anticipan estos problemas reales:
*   **Valores nulos por festivos:** MIBGAS no opera fines de semana ni festivos nacionales, dejando "huecos" que rompen la continuidad de la serie temporal estricta requerida por los modelos.
*   **Inconsistencia de separadores decimales:** Los CSV de Eurostat suelen venir con formato europeo (comas para decimales), lo cual genera errores al castear variables a `float` en Python si no se procesan antes.
*   **Datos extremos (*Outliers*):** Picos masivos de precio (ej. marzo de 2022) o precios eléctricos negativos por exceso renovable. No son errores, sino anomalías reales del mercado que los modelos lineales clásicos no sabrán interpretar sin regularización.

## 8. Decisiones de limpieza y transformación previstas
1.  **Tratamiento de nulos:** Se utilizará la técnica de *Forward-Fill* (rellenar con el último valor conocido) para los fines de semana, asumiendo que el precio del viernes se mantiene válido sábado y domingo.
2.  **Conversión de formatos:** Estandarización obligatoria de todo el texto (nombres de países en inglés) y *parsing* de strings a objetos `datetime`.
3.  **Variables derivadas (*Feature Engineering*):** Creación matemática de retardos (*Lags*), medias móviles, indicadores de volatilidad (desviación estándar a 7 días) y codificación cíclica temporal (seno/coseno para el día de la semana).
4.  **Codificación:** *One-Hot Encoding* para las variables categóricas espaciales (`Country`).

## 9. Riesgos del modelo de datos
*   **Parte más clara:** El flujo de los datos diarios del mercado MIBGAS. Es una estructura rígida y fácil de modelar.
*   **Parte con más incertidumbre:** El alineamiento exacto de eventos macroeconómicos (*dummies* geopolíticas) con respuestas instantáneas del mercado.
*   **Fuente más problemática:** Eurostat, debido a la posible discontinuidad o retraso en la publicación de los datos agregados semestrales para ciertos países del este.
*   **Alternativa de simplificación:** Si no se logra un modelo de datos robusto combinando Europa y España, el proyecto pivotará hacia un análisis univariante y exclusivamente centrado en la serie temporal del MIBGAS español, simplificando drásticamente el ensamblaje de la capa Gold y asegurando la entrega funcional.