# 01_ideas_producto.md

A continuación se exponen las ideas iniciales planteadas para el proyecto de Data Science e Inteligencia Artificial, incluyendo la idea final que fue seleccionada para el desarrollo definitivo del trabajo.

## 1. Análisis de datos deportivos
Esta primera idea consistiría en analizar datos relacionados con el ámbito futbolístico, como el porcentaje de pases acertados, tiros a puerta y otras métricas más complejas (progresión con y sin balón, participación en los goles de su equipo). La idea sería realizar una visualización de datos en las 5 grandes ligas (o profundizar en una de ellas en concreto) y, con esa información, entrenar un modelo de *Machine Learning* que nos ayude a predecir qué futbolistas jóvenes tienen o podrían tener una gran proyección en función de sus estadísticas.

## 2. Inversiones
En esta idea se planteó utilizar los datos de aumento/disminución del valor de las acciones de fondos de inversión para realizar una visualización sobre su evolución histórica. Adicionalmente, el objetivo sería entrenar un modelo de *Machine Learning* capaz de predecir el comportamiento de distintos fondos indexados en el corto, medio y largo plazo.

## 3. Optimización de entregas
Esta propuesta exploraba la posibilidad de utilizar los datos de reparto de empresas logísticas (como Amazon, CTT o Correos Express) para analizar sus trayectos diarios. El fin último sería entrenar un modelo de *Machine Learning* que permitiera optimizar las rutas de reparto, mejorando la eficiencia operativa.

## 4. Asistente Virtual para Ingeniería
Consistía en diseñar un sistema multiagente orientado a servir de apoyo en el trabajo diario de un Ingeniero Electrónico (diseño de placas, trabajo con componentes, etc.). La propuesta pasaba por utilizar datos de *datasheets*, libros de teoría y notas de aplicación para entrenar a un asistente virtual. Este sistema se integraría con las APIs de las grandes compañías distribuidoras para buscar componentes, resolver problemas de diseño de hardware y aclarar dudas teóricas.

---

## 5. Predicción del Mercado Energético (Proyecto Final Seleccionado)
Tras evaluar la viabilidad de las opciones anteriores, la idea finalmente seleccionada y desarrollada consiste en la creación de un sistema predictivo para el mercado de la electricidad y el gas natural. 

*   **Problema abordado:** La extrema volatilidad de los precios energéticos generada por crisis geopolíticas (Guerra de Ucrania, Oriente Medio) y factores macroeconómicos, lo cual dificulta la presupuestación y planificación operativa de las empresas.
*   **Enfoque metodológico:** Extracción, transformación y consolidación de datos procedentes de fuentes oficiales e institucionales (Eurostat y MIBGAS). Sobre esta capa de datos tratada, se entrena un ecosistema de modelos de *Machine Learning* (Ridge Regression, Random Forest, LightGBM) para predecir series temporales de precios.
*   **Valor del producto (MVP):** Un panel o sistema analítico que permite a comercializadoras, industrias electrointensivas y analistas logísticos visualizar el impacto del gas sobre la electricidad, anticipar sobrecostes a un mes vista y entender los motivos de las predicciones a través de herramientas de interpretabilidad (SHAP).