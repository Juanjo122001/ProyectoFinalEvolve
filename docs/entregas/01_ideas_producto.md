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
Tras evaluar la viabilidad de las opciones anteriores, la idea finalmente seleccionada y desarrollada consiste en el análisis del mercado energético europeo y la creación de un sistema predictivo para el gas natural. 

*   **Problema abordado:** La extrema volatilidad de los precios energéticos generada por crisis geopolíticas y macroeconómicas, lo cual dificulta la presupuestación y planificación operativa de las empresas.
*   **Enfoque metodológico:** El proyecto se divide funcionalmente en dos fases. Primero, un análisis exploratorio de la relación histórica entre el precio del gas y la electricidad en Europa. Una vez establecido este contexto, el proyecto se centra exclusivamente en predecir la serie temporal diaria del gas natural (MIBGAS) mediante algoritmos de regresión de *Machine Learning* (Ridge Regression, LightGBM).
*   **Valor del producto (MVP):** Un panel analítico que permite visualizar el impacto estructural del gas sobre la electricidad y anticipar el precio diario del gas a un mes vista mediante pronósticos recursivos, mejorando la toma de decisiones de compra en el mercado diario (*Spot*).