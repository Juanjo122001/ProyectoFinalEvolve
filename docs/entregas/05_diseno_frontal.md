# 05_diseno_frontal.md

## 1. Resumen de la solución y del usuario
*   **Problema:** La extrema volatilidad de los precios del gas natural (MIBGAS), que impide una planificación logística y presupuestaria fiable.
*   **Usuario principal:** Gestor de *trading* energético o analista de compras.
*   **Necesidad:** Anticipar la inercia del precio diario del gas a 30 días vista para optimizar estrategias operativas.
*   **Tipo de producto:** *Dashboard* analítico y predictor operativo.
*   **Acción principal:** Decidir si adelantar compras en el mercado *Spot* o aplicar estrategias de cobertura (*hedging*) basándose en la proyección técnica del mercado y la explicabilidad de sus factores.

## 2. Imagen mockup del frontal

![Mockup del frontal](../assets/05_mockup_frontal.jpg)

*(Nota de trazabilidad técnica: La imagen superior representa una visión conceptual a futuro del producto. Las especificaciones exactas sobre qué elementos visuales son viables con los datos actuales se detallan en la Sección 5).*

## 3. Justificación del diseño

### 3.1. Utilidad y valor de la solución
El *dashboard* centraliza la estimación del mercado a 30 días. Tras validar empíricamente que el MIBGAS se aproxima a un paseo aleatorio (*Random Walk*), el valor de la solución no reside en una predicción "mágica", sino en anclar las expectativas del negocio en una proyección inercial robusta (Modelo Naive) y advertir sobre los riesgos de degradación. Se prioriza mostrar la tendencia cruda y la explicabilidad del modelo (SHAP) para que el trader comprenda qué señales históricas a corto plazo están marcando el precio.

### 3.2. Flujo de usuario
1.  **Punto de entrada:** El usuario visualiza el último precio de cierre de MIBGAS y el KPI predictivo a 30 días.
2.  **Análisis de contexto:** Revisa la gráfica central para entender la evolución de las últimas semanas frente a la inercia proyectada.
3.  **Explicabilidad (Caja Blanca):** Consulta el panel lateral para identificar qué variables del *pipeline* (ej. el retardo del día anterior o la tendencia semanal) justifican el comportamiento actual del modelo, huyendo de las cajas negras.
4.  **Acción:** Ejecuta la estrategia de compra o asume una posición de espera, valorando si la tendencia a 30 días pone en riesgo su presupuesto.

### 3.3. Experiencia de usuario
*   **Jerarquía visual:** El precio actual y la métrica de predicción dominan la vista.
*   **Simplicidad y lenguaje de negocio:** Se omite deliberadamente la carga cognitiva de métricas estadísticas (MSE, RMSE, R²). El decisor consume exclusivamente euros por megavatio hora (€/MWh).
*   **Contexto:** La integración del histórico real empalmado con la línea de predicción permite una evaluación visual inmediata de la coherencia de la proyección.

## 4. Presentación de resultados y explicabilidad
El resultado principal es la estimación determinista del precio diario a 30 días vista. Para dotarlo de contexto transparente, el panel de "Influencia" alojará los resultados de interpretabilidad **SHAP**, mostrando el peso de las variables reales extraídas en la fase de ingeniería de características (retardos `Gas_Price_Lag_N`, promedios móviles temporales e indicadores). 

*Uso de IA Generativa:* No se implementará IA generativa. En un mercado crítico hipervolátil, el trader requiere trazabilidad matemática absoluta. Las explicaciones se basarán puramente en el análisis de valores de Shapley aplicados sobre los algoritmos.

## 5. Alcance real del MVP (Ajuste a la realidad analítica)
Para garantizar que el producto final promete exactamente lo que los datos y los modelos desarrollados pueden sostener matemáticamente, se establece una diferenciación estricta entre el diseño conceptual del *mockup* y el MVP funcional:

*   **Elementos implementados (MVP Funcional):** El frontal asumirá como línea base la predicción del **Modelo Naive** (el cual demostró en el *backtesting* recursivo ser la aproximación más segura a 30 días vista para mitigar la propagación del error). En paralelo, se integrará el análisis de impacto de variables (SHAP) aplicado sobre los modelos de *Machine Learning* (LightGBM/Ridge) utilizando exclusivamente las características temporales y de precios procesadas.
*   **Elementos conceptuales (Descartados del MVP actual):** El *mockup* visualiza módulos de "Niveles de almacenamiento de gas", "Bandas de confianza probabilística (94.1%)" y un panel de "Influencia eléctrica" desglosado por tecnologías (eólica, solar, nuclear, CO₂). **Estos elementos se marcan exclusivamente como conceptuales para futuras iteraciones del producto.** El *pipeline* actual de datos no captura estas variables exógenas y los algoritmos implementados son deterministas (no probabilísticos), por lo que mostrarlos supondría incurrir en métricas irreales.