# 04_analisis_modelado.md

**Nota de trazabilidad y actualización respecto a entregas anteriores:**
*Durante el diseño de esta Entrega 4, y tras refinar el alcance del proyecto, se ha tomado la decisión de separar el objetivo analítico del predictivo. La electricidad se utilizará exclusivamente para el análisis de contexto, mientras que la tarea de modelado se centrará al 100% en la predicción del gas natural (MIBGAS). Además, se descarta la regresión OLS como baseline en favor de un modelo Naive (referencia natural en forecasting) y se sustituye la validación estática (1-step ahead) por un backtesting recursivo a 30 días vista, para simular el uso real del MVP.*

## 1. Problema que se busca resolver

El mercado energético europeo atraviesa un escenario de alta incertidumbre debido a la dependencia del gas natural, la integración de energías renovables intermitentes y las crisis geopolíticas. Esto supone un problema crítico para las empresas comercializadoras y equipos de planificación logística, quienes no pueden optimizar sus compras de energía debido a la extrema volatilidad de los precios.

El proyecto producirá un **modelo predictivo (MVP)** que estimará de forma iterativa la evolución futura del precio diario del gas (MIBGAS) a 30 días vista. Este resultado será utilizado por gestores de *trading* energético para tomar decisiones estratégicas de cobertura de riesgos (*hedging*) y anticipar compras en el mercado *Spot*.

## 2. Análisis de datos planteado y utilidad esperada

El proyecto plantea un análisis en dos fases para extraer inteligencia de mercado directamente útil para el MVP:
*   **Fase 1 (Análisis de contexto eléctrico):** ¿Cuál es la magnitud estructural del impacto del precio del gas sobre la tarifa eléctrica en Europa? Se estudiará la relación histórica para demostrar por qué el gas es la variable crítica a predecir.
*   **Fase 2 (Análisis del mercado de gas):** Se estudiará la tendencia histórica, la volatilidad y la autocorrelación de los precios diarios de MIBGAS para validar la inercia temporal de la serie.
*   **Utilidad:** Este análisis proporcionará al usuario de negocio un contexto visual claro antes de consumir las predicciones algorítmicas, justificando por qué centramos el esfuerzo predictivo exclusivamente en el gas natural.

## 3. Tipo de modelos que se van a plantear

El proyecto aborda una tarea de **Forecasting de Series Temporales**. Dado que el MVP requiere proyectar 30 días hacia el futuro de forma recursiva, se evaluarán modelos de diferente naturaleza estadística.

| Alternativa | Tipo | Por qué se plantea | Limitación principal |
| :--- | :--- | :--- | :--- |
| **Baseline (Modelo Naive)** | Regla estadística simple | Es la referencia natural en *forecasting*. Asume que el precio de mañana será igual al de hoy ($t = t-1$) o al de hace una semana ($t = t-7$). | Incapaz de anticipar cambios de tendencia; reacciona a los movimientos del mercado con un retraso sistemático. |
| **Ridge Regression** | Lineal con penalización L2 | Estabiliza los pesos ante variables fuertemente correlacionadas. Es excelente para proyecciones suavizadas y resistentes a la acumulación de errores a largo plazo. | Incapacidad para modelar picos de volatilidad no lineales en el intradía. |
| **LightGBM** | Ensamblado *Gradient Boosting* | Su algoritmo *leaf-wise* es excepcionalmente rápido y preciso para capturar la inercia diaria a muy corto plazo. | Mayor riesgo de acumulación de error (*drift*) y predicciones erráticas en el *forecasting* iterativo a 30 días vista. |

## 4. Datos de entrada del análisis y los modelos

Los modelos predictivos consumirán directamente el fichero `gas_daily_prediction.csv`, mientras que el análisis de contexto se apoyará en `dataset_final.csv`.

| Entrada | Descripción | Granularidad / tipo | Uso en el análisis o modelo |
| :--- | :--- | :--- | :--- |
| **Dataset Gold** | `gas_daily_prediction.csv` | Una fila por día de negociación. | Fuente troncal unificada de variables de entrada ($X$) para la predicción. |
| **Trading_Day** | Fecha absoluta de la sesión. | Fecha (*Datetime*) | Excluida del entrenamiento (*Feature*) para evitar *Data Leakage*. Solo se usa como índice temporal. |
| **Gas_Price_Lag_N** | Variables retardadas (precio de hace 1, 7, 14 y 30 días). | Numérica (Derivada) | *Features* fundamentales. Inyectan memoria a corto y medio plazo al algoritmo. |
| **Gas_Rolling_Mean** | Medias y desviaciones móviles. | Numérica (Derivada) | *Feature* para que el modelo identifique la tendencia subyacente suavizando el ruido. |
| **Ukraine_War / Energy_Crisis** | Indicadores de contexto geopolítico. | Indicador (0/1) | Moduladores de cambio de régimen macroeconómico. |

## 5. Datos de salida y forma de consumo

La salida del modelo será consumida a través de un panel (*dashboard*) de análisis de negocio.

| Campo de salida | Descripción | Tipo | Uso posterior |
| :--- | :--- | :--- | :--- |
| **Prediction_Date** | Fechas correspondientes a la estimación (de $t+1$ hasta $t+30$). | *Datetime* | Representación en el eje X de la gráfica prospectiva del *dashboard*. |
| **Predicted_Price** | Predicción iterativa del precio de cierre del gas. | *Float* (€/MWh) | KPI central. Permite al gestor simular los costes del próximo mes. |
| **SHAP_Explanation** | Impacto de las variables que impulsan el modelo en el momento $t$. | *Gráfico de Barras* | Transparencia (XAI). Aumenta la confianza del decisor al explicar la lógica interna del precio estimado. |

## 6. Estrategia para diseñar y seleccionar el modelo

Para asegurar que el modelo seleccionado es útil y robusto, se seguirá esta estrategia:
1.  **Ingeniería rigurosa:** Creación de variables dinámicas (retardos y medias) que se recalcularán iterativamente en cada paso de la predicción futura.
2.  **Comparativa holística:** La selección del algoritmo ganador no dependerá de un R² inflado a un día vista, sino de su **estabilidad a largo plazo en predicciones recursivas**. 
3.  **Regla de decisión final:** Se seleccionará el modelo que logre acumular un menor error absoluto (MAE) al final del horizonte de 30 días y que presente una curva de proyecciones más realista, penalizando aquellos modelos que sufran sobreajuste severo o explosión de error.

## 7. Estrategia de validación y evaluación

La evaluación emulará de forma estricta un entorno de producción real (*out-of-sample*), evitando evaluar el modelo alimentándole datos reales del día anterior cuando se quiere predecir a un mes vista.

| Elemento | Decisión prevista | Justificación |
| :--- | :--- | :--- |
| **Separación de datos** | División temporal cronológica (Train 80% / Test 20%). | Obligatorio en series temporales para evitar fugas de información (*Data Leakage*). |
| **Validación Predictiva** | **Backtesting recursivo** con múltiples orígenes temporales en el Test. | Probar a 1 paso vista asumiendo que se conoce el valor real engaña las métricas. El backtesting obliga al modelo a alimentarse de sus propios errores pasados, demostrando su viabilidad real a 30 días. |
| **Métrica principal** | MAE a los 1, 7, 14 y 30 días. | Permite observar cómo se degrada el modelo a medida que se adentra en el futuro y medir el coste real del desvío en €/MWh. |
| **Baseline** | Modelo ingenuo (*Naive*: mañana será igual que hoy/hace 7 días). | Sirve como barrera de entrada. Si los algoritmos complejos no superan este modelo a lo largo de los 30 días, aportan complejidad sin valor. |

## 8. Riesgos y alternativas

*   **Riesgo de Data Leakage:** Es la principal amenaza. Se mitigará aplicando *sliding windows* rigurosas y asegurando que, durante la predicción a futuro, el modelo solo consuma las predicciones que él mismo acaba de generar, sin "espiar" el precio real.
*   **Degradación del error (*Drift*):** Al predecir 30 días de forma recursiva, un pequeño error en el día 1 se amplificará exponencialmente hacia el día 30.
*   **Alternativa táctica:** Si los modelos basados en árboles (LightGBM) demuestran incapacidad de generalización en el backtesting, la alternativa será priorizar la **Ridge Regression**. Al ser un modelo lineal con fuerte regularización L2, es menos susceptible a la explosión de error, garantizando una tendencia mucho más conservadora y útil para la presupuestación a largo plazo.