# 05_diseno_frontal.md

## 1. Resumen de la solución y del usuario
El mercado energético sufre una volatilidad extrema, dificultando la planificación financiera y logística. Para resolver este problema, el presente proyecto ha desarrollado un ecosistema de modelado predictivo de *Machine Learning* sobre el mercado eléctrico y el gas natural (MIBGAS).

*   **Usuario principal:** Analistas financieros, gestores de compras (departamentos de *trading*) y planificadores logísticos de empresas electrointensivas.
*   **Necesidad concreta:** Anticipar la tendencia de los precios de la energía a corto y medio plazo para presupuestar operaciones y decidir el momento óptimo de compra en el mercado *Spot* (diario).
*   **Tipo de producto:** Un **Dashboard Analítico y Predictor interactivo**.
*   **Resultado o acción principal:** El usuario obtendrá una curva visual con la predicción del precio a 30 días vista, acompañada de métricas explicativas. La acción principal derivada será ejecutar una orden de compra (si la predicción es alcista) o aplazar la compra (si la tendencia prevista es a la baja).

## 2. Imagen mockup del frontal

*(La siguiente imagen representa el boceto de la pantalla principal del MVP, orientada a la toma de decisiones del departamento de compras).*

![Mockup del frontal](../assets/05_mockup_frontal.png)

*Nota: La imagen muestra el selector de parámetros en la zona superior, la gráfica central combinando histórico y predicción (con intervalos de confianza), los KPIs numéricos destacados a la izquierda y el módulo de interpretabilidad SHAP a la derecha.*

## 3. Justificación del diseño

### 3.1. Utilidad y valor de la solución
El diseño del frontal no busca ser un mero repositorio de gráficos, sino un catalizador de decisiones de negocio. 
*   **Utilidad:** Transforma los complejos outputs matemáticos de los modelos *Ridge/LightGBM* en una interfaz limpia y procesable.
*   **Valor:** Ahorra horas de análisis estadístico manual y reduce el riesgo financiero de comprar energía en picos máximos. 
*   **Información esencial:** Se prioriza la curva futura (precio en €/MWh) y los factores influyentes, ocultando deliberadamente la complejidad algorítmica (hiperparámetros, métricas MAE/R²) para no abrumar al usuario de negocio.

### 3.2. Flujo de usuario
El recorrido del gestor de compras dentro de la herramienta es el siguiente:
1.  **Punto de entrada:** El usuario accede a la pantalla principal, que por defecto muestra el mercado MIBGAS en tiempo real (cotización actual).
2.  **Entradas o selecciones:** Utiliza los filtros superiores para seleccionar el horizonte de predicción (7, 14 o 30 días) y, opcionalmente, simular la activación de un "Cisne Negro" (ej. tensión geopolítica).
3.  **Procesamiento (oculto):** En el *backend*, los datos seleccionados se inyectan en el modelo entrenado, calculando las variables autorregresivas (lags) y generando el vector de predicciones.
4.  **Resultado:** El *dashboard* se actualiza. La línea continua (histórico) se engancha con una línea discontinua (predicción) que incluye bandas de confianza. A la derecha, el gráfico SHAP explica por qué el precio va a subir o bajar.
5.  **Acción:** Basado en la confianza que le genera la explicación gráfica, el gestor toma la decisión y pulsa en "Exportar Informe Predictivo" para adjuntarlo a su orden de compra.
6.  **Excepciones:** Si los datos de entrada (ej. caída de la web oficial de Eurostat) están corruptos, el sistema mostrará una alerta roja clara: "Alerta: Confianza del modelo baja por discontinuidad de datos en origen".

### 3.3. Experiencia de usuario
*   **Jerarquía visual:** El ojo va directamente a la curva central de precios, que ocupa el 60% de la pantalla, por ser el dato crítico.
*   **Simplicidad:** La paleta de colores es minimalista. Azul para datos reales y constatados, Naranja para estimaciones (creando una separación semántica clara entre certeza y probabilidad).
*   **Contexto y confianza:** Nunca se muestra la predicción como un valor exacto infalible. Se utiliza una franja o banda sombreada alrededor de la línea discontinua para indicar visualmente el margen de error del modelo, gestionando las expectativas.
*   **Control del usuario:** La visualización de los valores SHAP (por ejemplo, ver que una gran barra roja significa "fuerte inercia alcista del día anterior") otorga al usuario control cognitivo; entiende la "lógica" de la máquina y confía más en ella.

## 4. Presentación de resultados y explicabilidad
Para evitar el efecto de "caja negra" que genera rechazo en usuarios no técnicos, los resultados se comunican de forma transparente:
*   **Resultado principal:** Curva proyectada a 30 días y KPI numérico del precio promedio esperado en la próxima semana.
*   **Contexto (Incertidumbre):** Se acompaña el KPI con un rango (ej. "45.20 €/MWh ± 3.5 €"). Se evita presentar la estimación como certeza absoluta.
*   **Información técnica:** Todo el rigor matemático (*Loss curves*, curvas ROC, pesos exactos de Ridge) queda relegado a una pestaña secundaria llamada "Auditoría de Modelo", liberando el *dashboard* operativo.
*   **IA Generativa:** NO se utilizará IA Generativa (LLMs) para narrar o explicar los resultados. La explicabilidad del proyecto se apoya rigurosamente en el marco matemático SHAP (*SHapley Additive exPlanations*), garantizando una trazabilidad determinista y evitando el riesgo de "alucinaciones" al justificar decisiones financieras críticas.

## 5. Alcance del MVP
El diseño propuesto es ambicioso en su utilidad lógica, pero realista en cuanto a la tecnología implementable durante el curso.
*   **Implementado funcionalmente:** Se desarrollará el procesamiento de los datos subyacentes, la inferencia de los modelos (Ridge/LightGBM) y la generación estática de las gráficas (curvas de predicción y dependencias SHAP).
*   **Representación visual:** Elementos de interactividad compleja (como botones para cambiar el modelo en caliente, o alertas *push* al móvil) formarán parte únicamente del concepto visual (*mockup*), dado el tiempo disponible.
*   **Tecnología:** El *frontend* base operativo se construirá preferiblemente utilizando librerías ligeras de Python orientadas a datos, como `Streamlit` o `Dash`, permitiendo levantar la aplicación directamente sobre el *backend* de modelado ya desarrollado.