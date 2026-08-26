# 04_analisis_modelado.md

**Nota de trazabilidad y actualización respecto a entregas anteriores:**
*Durante el diseño de esta Entrega 4, se ha tomado la decisión arquitectónica de priorizar modelos lineales regularizados y de ensamblado (árboles), descartando otras aproximaciones como el Deep Learning. Esta modificación respecto a las ideas iniciales se justifica porque la serie temporal presenta cambios de régimen muy abruptos (Cisnes Negros geopolíticos) y un alto componente inercial, entornos donde modelos como Ridge o LightGBM ofrecen mayor robustez, menor riesgo de sobreajuste y permiten integrar interpretabilidad de negocio (SHAP).*

## 1. Problema que se busca resolver

El mercado energético europeo y español atraviesa un escenario de alta incertidumbre debido a la dependencia del gas natural, la integración de energías renovables intermitentes y las crisis geopolíticas (Guerra de Ucrania, tensiones en Oriente Medio). Esto supone un problema crítico para las empresas comercializadoras, las industrias electrointensivas y los equipos de planificación logística, quienes no pueden presupuestar sus operaciones ni optimizar sus compras de energía debido a la extrema volatilidad de los precios.

El proyecto producirá un **modelo predictivo (MVP)** que estimará la evolución futura de los precios de la electricidad y del gas (MIBGAS). Este resultado será utilizado por analistas financieros y gestores de *trading* energético para tomar decisiones estratégicas de cobertura de riesgos (*hedging*) y compra anticipada en el mercado *Spot*.

## 2. Análisis de datos planteado y utilidad esperada

Antes del modelado predictivo, el proyecto plantea un Análisis Exploratorio de Datos (EDA) para extraer inteligencia de mercado directamente útil para el MVP:
*   **Preguntas a responder:** ¿Cuál es la magnitud real del impacto del precio del gas sobre la tarifa eléctrica final? ¿Cómo reaccionan los distintos países europeos ante el mismo *shock* macroeconómico?
*   **Análisis a realizar:** Se estudiará la tendencia histórica, la estacionalidad intra-anual de la demanda y el impacto geográfico (clústeres de países). Se validará la autocorrelación de los precios diarios para comprobar si el mercado opera bajo un "paseo aleatorio" (*random walk*).
*   **Utilidad:** Este análisis no será un mero trámite; se integrará en la visualización inicial del MVP para proporcionar contexto al usuario (por ejemplo, mostrando las medias móviles recientes) y servirá para seleccionar las características (*feature selection*) clave para los algoritmos.

## 3. Tipo de modelos que se van a plantear

El proyecto aborda una tarea de **Regresión y Forecasting de Series Temporales**. Se probarán familias de modelos que ofrezcan un buen equilibrio entre precisión y explicabilidad.

| Alternativa | Tipo | Por qué se plantea | Limitación principal |
| :--- | :--- | :--- | :--- |
| **Baseline (Regresión OLS)** | Modelo lineal simple | Proporciona la referencia mínima de rendimiento frente a la que comparar. | Muy vulnerable a la multicolinealidad estructural (fechas vs. geopolítica), lo que genera coeficientes inestables. |
| **Ridge Regression** | Lineal con penalización L2 | Estabiliza los pesos ante variables fuertemente correlacionadas. Es excelente para proyecciones suavizadas y estables a largo plazo (30 días). | Incapacidad para modelar picos de volatilidad no lineales en el intradía. |
| **Random Forest** | Ensamblado *Bagging* | Esencial para modelar la electricidad por su capacidad nativa de manejar las variables categóricas (los distintos países) sin sesgos de escala. | Como modelo de árboles, no puede extrapolar numéricamente tendencias futuras que caigan fuera del dominio de entrenamiento. |
| **LightGBM** | Ensamblado *Gradient Boosting* | Su algoritmo *leaf-wise* es excepcionalmente rápido y preciso para capturar la inercia diaria del gas natural. | Mayor riesgo de acumulación de error (*drift*) en el *forecasting* iterativo a varios días vista. |

## 4. Datos de entrada del análisis y los modelos

Los modelos consumirán directamente los ficheros de la capa Gold (`gas_daily_prediction.csv` y `dataset_final.csv`) definidos en la Entrega 3.

| Entrada | Descripción | Granularidad / tipo | Uso en el análisis o modelo |
| :--- | :--- | :--- | :--- |
| **Dataset Gold** | `gas_daily_prediction.csv` (principal para Forecasting) y `dataset_final.csv`. | Una fila por día de negociación / país y semestre. | Fuente troncal unificada de variables de entrada (X). |
| **Trading_Day** | Fecha absoluta de la sesión de mercado. | Fecha (*Datetime*) | Excluida del entrenamiento (*Feature*) para evitar *Data Leakage*. Solo usada para indexar. |
| **Gas_Price_Lag_N** | Variables retardadas (precio de hace 1, 7 y 30 días). | Numérica (Derivada) | *Features* fundamentales. Inyectan memoria a corto plazo al algoritmo. |
| **Gas_Rolling_Mean** | Medias móviles semanales para suavizar el ruido. | Numérica (Derivada) | *Feature* para que el modelo identifique la tendencia subyacente. |
| **Country_*** | Variables *dummy* espaciales. | Categórica (One-Hot) | *Feature* para el modelo de predicción eléctrica europea (Random Forest). |
| **Ukraine_War** | Indicador de conflicto activo. | Indicador (0/1) | Variable de contexto para marcar el cambio de régimen macroeconómico. |

## 5. Datos de salida y forma de consumo

La salida del modelo debe ser altamente interpretable y se integrará en un panel (*dashboard*) de análisis de negocio.

| Campo de salida | Descripción | Tipo | Uso posterior |
| :--- | :--- | :--- | :--- |
| **Market_ID / Country** | Identificador del mercado (ej. MIBGAS, ES, DE). | *String* | Trazabilidad y filtrado interactivo en el MVP. |
| **Prediction_Date** | Fecha correspondiente a la estimación (ej. t+1). | *Datetime* | Representación en el eje temporal del gráfico del *dashboard*. |
| **Predicted_Price** | Predicción del precio de cierre de la energía. | *Float* (€/MWh) | KPI central. Permite al gestor decidir si compra o espera. |
| **SHAP_Explanation** | Impacto de las variables que empujan esa predicción (ej. "Alta inercia del día anterior"). | *Texto / JSON* | Transparencia (XAI). Aumenta la confianza del decisor al explicar el motivo del precio estimado. |

## 6. Estrategia para diseñar y seleccionar el modelo

Para asegurar que el modelo seleccionado es útil y robusto, se seguirá esta estrategia:
1.  **Ingeniería cuidadosa:** Creación de variables de *momentum* y componentes cíclicos sin generar *Data Leakage* (la fila $t$ solo verá datos hasta $t-1$).
2.  **Preprocesamiento dual:** Se aislará el flujo de datos. Para Ridge Regression se aplicará `StandardScaler`; para LightGBM y Random Forest se inyectarán los datos sin escalar.
3.  **Comparativa holística:** La selección del algoritmo ganador no dependerá solo de la minimización del error absoluto (MAE), sino de su **estabilidad a largo plazo** en predicciones recursivas y su **interpretabilidad**. Se penalizarán los modelos que, aunque ligeramente más precisos a 1 día vista, sufran sobreajuste severo o explosión de error a 30 días.

## 7. Estrategia de validación y evaluación

La evaluación emulará un entorno de producción real, respetando estrictamente la flecha del tiempo.

| Elemento | Decisión prevista | Justificación |
| :--- | :--- | :--- |
| **Separación de datos** | División temporal (*Temporal Train-Test Split*). 80% histórico, 20% futuro. | Obligatorio en series temporales para evitar fugas de información. La validación aleatoria cruzaría datos futuros hacia el entrenamiento. |
| **Métrica principal 1** | MAE (Error Absoluto Medio). | Expresa el error medio de desvío en €/MWh, siendo la métrica más directa y útil para calcular el coste económico del error en un presupuesto. |
| **Métrica principal 2** | R² (Coeficiente de Determinación). | Indica el porcentaje de varianza del mercado que el modelo es capaz de explicar. Útil para comparar arquitecturas diferentes. |
| **Baseline** | Regresión Lineal Clásica (OLS). | Sirve como barrera de entrada. Si un modelo avanzado no supera la robustez del *baseline* OLS, se descarta. |
| **Criterio de aceptación** | Superar las métricas del *baseline* y mantener coherencia en las explicaciones de los valores SHAP. | Garantiza un modelo matemáticamente superior y lógicamente comprensible. |

## 8. Riesgos y alternativas

*   **Riesgo de Data Leakage:** Es la amenaza principal en el modelado temporal. Se mitigará eliminando explícitamente la variable absoluta de fecha y aplicando *sliding windows* rigurosas para el cálculo de retardos.
*   **"Cisnes Negros" e hipervolatilidad:** Los modelos son susceptibles a fallar si estalla una nueva crisis no representada en el histórico de entrenamiento.
*   **Alternativa táctica:** Si los modelos avanzados (Random Forest / LightGBM) demuestran incapacidad de generalización (*overfitting*) sobre el conjunto de test, el sistema recaerá (*fallback*) en el uso de **Ridge Regression**. Al estar fuertemente regularizado, sacrificará agilidad en el intradía a cambio de proporcionar una banda de precios tendencial mucho más conservadora y resistente a *shocks* erráticos.