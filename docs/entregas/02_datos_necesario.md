# 02_datos_necesarios.md

## 1. Idea seleccionada

**Párrafo 1 - Problema que resuelve:** El mercado energético europeo y español atraviesa un periodo de volatilidad histórica debido a la dependencia de los hidrocarburos, la transición ecológica y tensiones geopolíticas (guerra de Ucrania, Oriente Medio). Esta inestabilidad supone un problema crítico para las comercializadoras de energía, la industria electrointensiva y la planificación logística, que se enfrentan a sobrecostes imprevistos e incapacidad para presupuestar operaciones con fiabilidad. Resolver esto aportaría un enorme valor económico al permitir estrategias de cobertura (*hedging*) basadas en datos empíricos y no en intuición.

**Párrafo 2 - Solución planteada:** La solución propuesta consiste en aplicar técnicas de *Data Science* y *Machine Learning* para analizar y predecir estos precios. Se desarrollarán algoritmos paramétricos (Ridge Regression) y basados en árboles de decisión (Random Forest, LightGBM) para, en una primera fase, modelar el impacto del gas sobre la electricidad en Europa; y en una segunda fase, realizar un *forecasting* recursivo (a 30 días vista) del precio diario del gas natural. Además, se aplicarán técnicas de explicabilidad (SHAP) para entender el "por qué" de las predicciones.

**Párrafo 3 - MVP del proyecto final:** El Producto Mínimo Viable (MVP) consistirá en un *dashboard* analítico o informe automatizado que mostrará tres elementos clave: 1) Análisis visual de la evolución histórica y geográfica de los precios. 2) La curva de predicción a futuro del precio de la energía. 3) Gráficos de interpretabilidad (SHAP) que justifiquen a los usuarios de negocio qué factores (inercia, eventos geopolíticos) están empujando la predicción al alza o a la baja en ese momento.

## 2. Datos necesarios

Para desarrollar el modelo predictivo con rigor, se necesita:
*   **Variables:** Precio de cierre de la electricidad, precio diario del gas, fechas, país, e indicadores booleanos de eventos geopolíticos.
*   **Granularidad:** Para el análisis europeo, granularidad semestral por país. Para el modelo principal del gas, **granularidad diaria** por sesión de mercado.
*   **Profundidad histórica:** Datos desde principios de 2021 hasta la actualidad. Esto es crítico porque abarca un periodo "normal" pre-crisis, el pico de la guerra, y la posterior estabilización.
*   **Volumen:** Se estiman unos ~2000 registros diarios para el mercado del gas y varios cientos para el histórico europeo. Es un volumen tabular manejable e ideal para algoritmos clásicos de *Machine Learning*.
*   **Imprescindibles vs. Deseables:** Es imprescindible el histórico de precios (MIBGAS/Eurostat) y las fechas. Sería deseable (pero no obligatorio) incluir variables meteorológicas (temperatura media) para medir la demanda por frío/calor, aunque se priorizará la componente inercial de la serie temporal.

## 3. Fuentes de datos previstas

*   **Fuentes concretas:** Eurostat (Oficina Estadística de la UE) y MIBGAS (Mercado Ibérico del Gas).
*   **Accesibilidad:** Son fuentes 100% abiertas, públicas y gratuitas.
*   **Enlaces:** [MIBGAS Data](https://www.mibgas.es/) | [Eurostat Database](https://ec.europa.eu/eurostat/data/database)
*   **Formato:** Descarga directa en ficheros CSV y Excel.
*   **Histórico y estabilidad:** Cuentan con un histórico de más de una década. Son instituciones oficiales, por lo que su estabilidad, mantenimiento y rigor son altísimos.
*   **Riesgos:** El principal riesgo es la existencia de días sin cotización (fines de semana o festivos) que generarán "huecos" temporales (valores nulos) en la serie temporal y obligarán a aplicar técnicas de imputación.

## 4. Consideraciones de privacidad y protección de datos

El proyecto utiliza exclusivamente datos macroeconómicos e índices de mercado público. 
*   **Ausencia de PII:** No existe absolutamente ninguna Información Personal Identificable (PII). No hay datos de clientes, usuarios ni facturación privada.
*   **Seguridad:** Su uso es 100% seguro y ético para un entorno académico. No están sujetos al RGPD, no requieren anonimización y no presentan ningún conflicto de confidencialidad.

## 5. Viabilidad inicial del proyecto

*   **¿Parece viable obtener los datos?** Sí, la viabilidad es absoluta al tratarse de portales institucionales de descarga directa.
*   **¿Tienen calidad e histórico?** Sí, al ser datos financieros auditados, su calidad es superior a la media de otros conjuntos de datos web.
*   **¿Es realista para el curso?** Sí. Al no requerir infraestructuras *Big Data* masivas (como imágenes o texto), el esfuerzo se puede centrar en el modelado algorítmico y matemático (Machine Learning).
*   **Riesgo principal:** El mayor riesgo algorítmico es que los *shocks* geopolíticos son impredecibles (Cisnes Negros). Los modelos podrían sufrir para extrapolar el futuro si ocurre un evento no visto en el entrenamiento.
*   **Alternativa:** Si las fuentes fallan o los modelos complejos se sobreajustan, la alternativa será simplificar el *scope* utilizando un modelo predictivo lineal muy regularizado (Ridge) o modelos puramente estadísticos (ARIMA) sobre un único mercado, garantizando la entrega del MVP.