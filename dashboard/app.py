"""
VOLT · Energy Intelligence
Dashboard del TFM «Análisis y predicción de precios en los mercados
energéticos de electricidad y gas natural» — Juan José Aguilera Conde.

Ejecutar desde la raíz del proyecto:
    python dashboard/app.py
y abrir http://127.0.0.1:8050
"""

from __future__ import annotations

import dash
from dash import Dash, Input, Output, State, dcc, html, no_update

import charts as C
import data as D
from theme import MODEL_LABEL

# ------------------------------------------------------------------
# Utilidades de formato (estilo español)
# ------------------------------------------------------------------
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def fnum(x: float, d: int = 2) -> str:
    s = f"{x:,.{d}f}"
    return s.replace(",", "·").replace(".", ",").replace("·", ".")


def fdate(ts) -> str:
    return f"{ts.day:02d} {MESES[ts.month - 1]} {ts.year}"


def pct(x: float, d: int = 1) -> str:
    return ("+" if x > 0 else "") + fnum(x * 100, d) + " %"


GRAPH_CONFIG = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines",
                               "hoverClosestCartesian", "hoverCompareCartesian"],
    "toImageButtonOptions": {"format": "png", "scale": 3},
}


def graph(fig=None, gid=None, **kw):
    props = {"config": GRAPH_CONFIG}
    if gid:
        props["id"] = gid
    if fig is not None:
        props["figure"] = fig
    props.update(kw)
    return dcc.Graph(**props)


def icon(name: str, cls: str = ""):
    return html.Span(className=f"ico i-{name} {cls}")


def card(children, title=None, sub=None, foot=None, cls="", right=None):
    head = []
    if title:
        head = [html.Div([
            html.Div([html.Div(title, className="card-title"),
                      html.Div(sub, className="card-sub") if sub else None]),
            right,
        ], className="card-head")]
    return html.Div(head + (children if isinstance(children, list) else [children])
                    + ([html.Div(foot, className="card-foot")] if foot else []),
                    className=f"card {cls}")


def kpi(label, value, unit="", note=None, accent=None, text=False):
    return html.Div([
        html.Span(className=f"kpi-accent bg-{accent}") if accent else None,
        html.Div(label, className="kpi-label"),
        html.Div([value, html.Span(unit, className="kpi-unit") if unit else None],
                 className="kpi-value text" if text else "kpi-value"),
        html.Div(note, className="kpi-note") if note else None,
    ], className="card kpi")


def insight(children, gas=False):
    return html.Div([icon("info"), html.Div(children)], className="insight gas" if gas else "insight")


def slider(id_, **kw):
    """dcc.Slider compatible con Dash 2, 3 y 4 (en Dash 4 oculta el input numérico)."""
    if "allow_direct_input" in getattr(dcc.Slider, "_prop_names", []):
        kw["allow_direct_input"] = False
    return dcc.Slider(id=id_, **kw)


ES_LABELS = {"select_all": "Seleccionar todo", "deselect_all": "Quitar todo", "selected_count": "{num} seleccionados",
             "search": "Buscar", "clear_search": "Borrar búsqueda", "clear_selection": "Borrar selección",
             "no_options_found": "Sin resultados"}


def dropdown(id_, **kw):
    """dcc.Dropdown con textos en español cuando la versión de Dash lo permite."""
    if "labels" in getattr(dcc.Dropdown, "_prop_names", []):
        kw["labels"] = ES_LABELS
    return dcc.Dropdown(id=id_, **kw)


def seg(id_, options, value):
    return dcc.RadioItems(id=id_, options=options, value=value, className="seg", inline=True)


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
D.warmup()
H = D.headline()

app = Dash(
    __name__,
    title="VOLT · Energy Intelligence | TFM",
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;650;700&display=swap"
    ],
)
server = app.server

PAGES = [
    ("Análisis", [
        ("/", "Resumen", "home", "Visión general del proyecto"),
        ("/mercado", "Mercado europeo", "globe", "Precio de la electricidad doméstica en 40 países"),
        ("/relacion", "Gas ↔ Electricidad", "link", "Cómo el gas fija el precio de la luz en la Península"),
    ]),
    ("Modelado", [
        ("/modelos", "Modelos de electricidad", "bars", "Comparativa de algoritmos sobre Eurostat + MIBGAS"),
        ("/prediccion", "Predicción del gas", "trend", "Proyección recursiva a 30 días del precio MIBGAS"),
        ("/shap", "Explicabilidad", "spark", "Qué variables mueven cada predicción (SHAP)"),
    ]),
    ("Proyecto", [
        ("/metodologia", "Metodología", "layers", "Datos, variables, validación y trabajo futuro"),
    ]),
]
PAGE_META = {p: (name, desc) for _, items in PAGES for p, name, _, desc in items}


def sidebar():
    nav = []
    for section, items in PAGES:
        nav.append(html.Div(section, className="nav-section"))
        for path, name, ico, _ in items:
            nav.append(dcc.Link([icon(ico), html.Span(name)], href=path, id=f"nav-{path.strip('/') or 'home'}",
                                className="nav-link"))
    return html.Aside([
        html.Div([
            html.Div(icon("bolt"), className="brand-mark"),
            html.Div([html.Div("VOLT", className="brand-name"),
                      html.Div("Energy Intelligence", className="brand-sub")]),
        ], className="brand"),
        html.Nav(nav),
        html.Div([
            html.Div([
                html.Div("JA", className="avatar"),
                html.Div([html.Div("Juan José Aguilera Conde", className="author-name"),
                          html.Div("TFM · Máster IA · 2025–26", className="author-role")]),
            ], className="author"),
            html.Button([icon("sun", ""), html.Span("Cambiar tema", id="theme-label")],
                        id="theme-btn", className="theme-toggle", n_clicks=0),
        ], className="sidebar-foot"),
    ], className="sidebar")


def topbar():
    return html.Header([
        html.Div([html.Div(id="page-eyebrow", className="topbar-eyebrow"),
                  html.Div(id="page-title", className="topbar-title")]),
        html.Div([
            html.Span([html.Span(className="dot", style={"background": "var(--orange)"}),
                       f"MIBGAS hasta {fdate(H['last_date'])}"], className="chip"),
            html.Span([icon("cal"), "Eurostat 2021-S1 → 2025-S2"], className="chip"),
            html.Span(f"{H['n_countries']} países · {fnum(H['n_days'], 0)} días", className="chip"),
        ], className="chips"),
    ], className="topbar")


app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="theme", storage_type="local", data="dark"),
    sidebar(),
    html.Main([topbar(), dcc.Loading(html.Div(id="page", className="content"), type="dot",
                                     color="#3987e5", delay_show=250)], className="main"),
], className="shell")


# ==================================================================
# PÁGINAS
# ==================================================================
def page_home(theme):
    bt, _ = D.backtest()
    r30 = bt[bt["Horizon"] == 30].set_index("Model")["RMSE"]
    wk = H["week_change"]
    findings = [
        ("El gas marca el precio de la luz",
         f"ρ = {fnum(H['spearman'])}",
         "Correlación de Spearman entre gas MIBGAS y electricidad doméstica en España y Portugal. "
         "La guerra de Ucrania y la crisis energética alcanzan ρ ≈ 0,70."),
        ("Europa no es un mercado único",
         "9.º más caro",
         "España ocupa el puesto 9 de 40. Bélgica, Irlanda y Alemania —muy dependientes del gas— "
         "encabezan el ranking; Balcanes y Europa del Este cierran la tabla."),
        ("Random Forest explica el precio eléctrico",
         f"R² = {fnum(H['rf_r2'], 3)}",
         f"Mejor modelo sobre el último año (MAE {fnum(H['rf_mae'], 4)} €/kWh). La identidad del país pesa "
         "más que el propio gas: el mercado sigue fragmentado."),
        ("El mercado del gas es eficiente a corto plazo",
         "t-1 ≈ 63 %",
         "En LightGBM, el precio del día anterior concentra casi dos tercios de la importancia SHAP: "
         "el mercado incorpora la información casi al instante."),
        ("A 30 días, la inercia gana a la complejidad",
         f"{fnum(r30['Naive'])} vs {fnum(r30['LightGBM'])}",
         "RMSE (€/MWh) del modelo Naïve frente a LightGBM en backtesting recursivo. El error de los modelos "
         "de ML se acumula al alimentarse de sus propias predicciones."),
    ]
    return html.Div([
        html.Section([
            html.Span([html.Span(className="dot", style={"background": "var(--orange)"}),
                       "Trabajo Fin de Máster · Máster en Inteligencia Artificial"], className="hero-eyebrow"),
            html.H1(["Análisis y predicción de precios en los mercados de ",
                     html.Span("electricidad", className="hl-blue"), " y ",
                     html.Span("gas natural", className="hl-orange")]),
            html.P("¿Cuánto del precio de la luz en Europa se explica por el gas? ¿Podemos anticipar el precio "
                   "diario del gas ibérico? Integramos datos oficiales de Eurostat y MIBGAS, los analizamos a la luz "
                   "de las crisis geopolíticas de 2021–2026 y comparamos modelos de aprendizaje automático, "
                   "interpretados con SHAP.", className="hero-lead"),
            html.Div([
                html.Div([html.B("Eurostat"), "Electricidad doméstica · semestral"]),
                html.Div([html.B("MIBGAS"), "Gas PVB · diario (€/MWh)"]),
                html.Div([html.B("8 algoritmos"), "Lineales, árboles, boosting y redes"]),
                html.Div([html.B("SHAP"), "Interpretabilidad de cada predicción"]),
            ], className="hero-meta"),
        ], className="hero"),

        html.Div([
            kpi("Último precio MIBGAS", fnum(H["last_price"]), "€/MWh",
                [html.Span(pct(wk), className=f"delta {'up' if wk > 0 else 'down'}"),
                 f" en 7 días · {fdate(H['last_date'])}"], accent="orange"),
            kpi("Correlación gas–electricidad", fnum(H["spearman"]), "ρ",
                "Spearman · España y Portugal · 2021–2025", accent="blue"),
            kpi("Precisión modelo eléctrico", fnum(H["rf_r2"], 3), "R²",
                f"Random Forest · MAE {fnum(H['rf_mae'], 4)} €/kWh", accent="blue"),
            kpi("Máximo histórico del gas", fnum(H["peak_price"], 0), "€/MWh",
                f"{fdate(H['peak_date'])} · 5× la mediana del periodo", accent="orange"),
        ], className="grid g-4"),

        card([graph(C.gas_history(theme))],
             title="Precio diario del gas natural en el mercado ibérico (MIBGAS, PVB)",
             sub=f"{fdate(H['gas_start'])} – {fdate(H['gas_end'])} · franjas sombreadas: periodos de crisis; "
                 "líneas: inicio de conflictos",
             foot="Fuente: elaboración propia a partir de MIBGAS (2026). Producto diario GDAES D+1."),

        html.Div("Hallazgos principales", className="section-title"),
        html.Div([
            html.Div([
                html.Div(f"0{i + 1}", className="finding-num"),
                html.Div(metric, className="finding metric num"),
                html.H3(title),
                html.P(text),
            ], className="card finding") for i, (title, metric, text) in enumerate(findings[:3])
        ], className="grid g-3"),
        html.Div([
            html.Div([
                html.Div(f"0{i + 4}", className="finding-num"),
                html.Div(metric, className="finding metric num"),
                html.H3(title),
                html.P(text),
            ], className="card finding") for i, (title, metric, text) in enumerate(findings[3:])
        ], className="grid g-2"),

        html.Div("Contexto geopolítico del periodo de estudio", className="section-title"),
        card(html.Div([
            html.Div([
                html.Div(className="tl-dot"),
                html.Div(fdate_range(ev), className="tl-date"),
                html.Div(ev["label"], className="tl-title"),
                html.Div(ev["text"], className="tl-text"),
            ], className="tl-item") for ev in D.EVENTS
        ], className="timeline")),
    ])


def fdate_range(ev):
    import pandas as pd
    s = fdate(pd.Timestamp(ev["start"]))
    return f"{s} → {fdate(pd.Timestamp(ev['end']))}" if ev["end"] else f"Desde {s}"


TAX_OPTS = [{"label": v, "value": k} for k, v in D.TAX_OPTIONS.items()]
COUNTRY_OPTS = sorted([{"label": D.COUNTRY_ES[c], "value": c} for c in D.COUNTRY_ES], key=lambda o: o["label"])
SEMS = D.semesters()
SEM_MARKS = {0: "Media"} | {i + 1: s.replace("20", "", 1) for i, s in enumerate(SEMS)}


def page_market(theme):
    return html.Div([
        html.Div([
            html.Div([html.Span("Modalidad de impuestos", className="control-label"),
                      dropdown("mk-tax", options=TAX_OPTS, value="all", clearable=False,
                                   searchable=False, style={"minWidth": 280})], className="control"),
            html.Div([html.Span("Países a comparar (máx. 4)", className="control-label"),
                      dropdown("mk-countries", options=COUNTRY_OPTS, value=["Spain", "Germany"],
                                   multi=True, clearable=False)], className="control grow"),
        ], className="controls"),

        html.Div(id="mk-kpis", className="grid g-4"),

        html.Div([
            card([graph(gid="mk-europe")], title="Evolución semestral frente a la media europea",
                 sub="Precio medio de la electricidad doméstica (banda DC, 2.500–4.999 kWh/año)"),
            card([graph(gid="mk-iberia")], title="España y Portugal frente al gas ibérico",
                 sub="Todo en €/kWh: el gas MIBGAS se convierte y promedia por semestre"),
        ], className="grid g-2"),

        html.Div([
            card([
                html.Div([html.Span("Periodo", className="control-label"),
                          slider("mk-sem", min=0, max=len(SEMS), step=1, value=0, marks=SEM_MARKS,
                                     included=False)], style={"marginBottom": "6px"}),
                graph(gid="mk-map", config={**GRAPH_CONFIG, "topojsonURL": app.get_asset_url("topojson/")}),
            ], title="Mapa de precios", sub="Precio medio por país en el periodo seleccionado · Kosovo no se representa en el mapa"),
            card([graph(gid="mk-rank")], title="Ranking de países", sub="España y Portugal resaltados"),
        ], className="grid g-7-5"),

        html.Div([
            card([graph(gid="mk-box")], title="Dispersión en los 15 países más caros",
                 sub="Distribución de todas las observaciones 2021–2025"),
            card([graph(gid="mk-hist-e")], title="Distribución del precio eléctrico",
                 sub="Cola larga hasta 0,6 €/kWh: el rastro de la crisis de 2022"),
        ], className="grid g-2"),
        insight([html.B("Lectura: "), "entre 2021 y 2022 España se separa de la media europea y supera los 0,30 €/kWh "
                 "porque su tarifa regulada estaba indexada al mayorista. La Excepción Ibérica (junio 2022) "
                 "provoca una caída brusca mientras Europa alcanza su máximo. Desde 2023 los precios se "
                 "estabilizan, aunque España sigue por encima de la media."]),
    ])


def page_relation(theme):
    return html.Div([
        html.Div([
            html.Div([html.Span("Modalidad de impuestos", className="control-label"),
                      dropdown("rl-tax", options=TAX_OPTS, value="all", clearable=False,
                                   searchable=False, style={"minWidth": 280})], className="control"),
        ], className="controls"),
        html.Div(id="rl-kpis", className="grid g-3"),
        html.Div([
            card([graph(gid="rl-heat")], title="Matriz de correlación de Spearman",
                 sub="Mercado ibérico (España + Portugal), agregación semestral 2021–2025",
                 foot="Rojo: relación positiva · azul: negativa · gris: sin relación."),
            card([graph(gid="rl-scatter")], title="Gas frente a electricidad, semestre a semestre",
                 sub="Cada punto es un semestre · línea de puntos: tendencia lineal orientativa"),
        ], className="grid g-2"),
        html.Div([
            card(html.Div(id="rl-table"), title="Estadísticos descriptivos por semestre",
                 sub="Media, mediana y desviación estándar (€/kWh) · Tablas 1 y 2 de la memoria"),
            html.Div([
                insight([html.B("Vínculo fuerte: "), "ρ = 0,61 entre gas y electricidad confirma estadísticamente "
                         "que ambos mercados están estrechamente ligados, coherente con un mercado marginalista donde "
                         "las centrales de gas fijan el precio ~39 % de las horas (Zakeri et al., 2023)."]),
                html.Div(style={"height": "12px"}),
                insight([html.B("Sensibilidad geopolítica: "), "Guerra de Ucrania (0,70) y Crisis energética (0,71) "
                         "muestran que los precios domésticos ibéricos reaccionaron con fuerza a estos shocks."], gas=True),
                html.Div(style={"height": "12px"}),
                insight([html.B("Oriente Medio aún no aparece: "), "la correlación negativa refleja que, con los "
                         "datos semestrales disponibles hasta 2025-S2, su efecto todavía no se observa."]),
            ]),
        ], className="grid g-7-5"),
    ])


def page_models(theme):
    m = D.electricity_model_results()
    best = m.iloc[0]
    rows = []
    for _, r in m.iterrows():
        lin = r["Model"] == "Linear Regression"
        cls = "best" if r["Model"] == best["Model"] else ("dim" if lin else "")
        rows.append(html.Tr([
            html.Td([r["Model"], html.Span("mejor", className="badge ok") if cls == "best" else None,
                     html.Span("colapsa", className="badge warn") if lin or r["R2"] < 0 else None], className="name"),
            html.Td("≈10²¹" if lin else fnum(r["MSE"], 5)),
            html.Td("≈3·10¹⁰" if lin else fnum(r["RMSE"], 4)),
            html.Td("≈3·10¹⁰" if lin else fnum(r["MAE"], 4)),
            html.Td("−1,5·10²³" if lin else fnum(r["R2"], 3)),
        ], className=cls))
    table = html.Table([html.Thead(html.Tr([html.Th("Modelo"), html.Th("MSE"), html.Th("RMSE"),
                                            html.Th("MAE"), html.Th("R²")])), html.Tbody(rows)], className="tbl")
    return html.Div([
        html.Div([
            kpi("Mejor modelo", "Random Forest", "", "300 árboles · sin escalado · partición cronológica", "blue",
                text=True),
            kpi("Varianza explicada", fnum(best["R2"], 3), "R²", "Test: último año disponible (2025)", "blue"),
            kpi("Error absoluto medio", fnum(best["MAE"], 4), "€/kWh", f"RMSE {fnum(best['RMSE'], 4)} €/kWh", "blue"),
            kpi("Red neuronal (MLP)", fnum(m.set_index('Model').loc['MLP', 'R2'], 2), "R²",
                "Peor que predecir la media: muy pocos datos", "orange"),
        ], className="grid g-4"),
        html.Div([
            card([graph(C.electricity_r2(theme))], title="Coeficiente de determinación por modelo",
                 sub="Regresión lineal excluida del gráfico (R² ≈ −1,5·10²³ por multicolinealidad)"),
            card([table], title="Métricas en el conjunto de test", sub="Tabla 6 de la memoria · unidades en €/kWh"),
        ], className="grid g-5-7"),
        html.Div([
            card([graph(C.rf_importance(theme))], title="Importancia de variables · Random Forest",
                 sub="Top 15 variables, coloreadas por familia"),
            card([
                graph(C.rf_group_share(theme)),
                html.Div(style={"height": "14px"}),
                insight([html.B("El país pesa más que el gas. "), "Bélgica, Dinamarca, Irlanda y Alemania encabezan la "
                         "importancia: el modelo aprende un «precio base» por país. Diferencias de mix, fiscalidad y "
                         "ayudas públicas impiden la convergencia del mercado europeo."]),
                html.Div(style={"height": "12px"}),
                insight([html.B("Multicolinealidad. "), "Guerra, crisis y año avanzan juntos en el tiempo; la "
                         "regresión lineal dispara sus coeficientes (~10¹⁶) y Ridge (L2) los estabiliza hasta R² = 0,60."],
                        gas=True),
                html.Div(style={"height": "12px"}),
                insight([html.B("Por qué pasar al gas diario. "), "Con datos semestrales (1.164 observaciones) el problema "
                         "eléctrico queda limitado; MIBGAS ofrece 1.949 días, suficientes para un modelo predictivo diario."]),
            ], title="Importancia agregada por familia", sub="Suma de la importancia de cada grupo de variables"),
        ], className="grid g-7-5"),
    ])


def page_forecast(theme):
    return html.Div([
        html.Div([
            html.Div([html.Span("Modelos", className="control-label"),
                      dcc.Checklist(id="fc-models", className="pills", inline=True,
                                    value=["Naive", "LightGBM", "Ridge Regression"],
                                    options=[{"label": html.Span("LightGBM"), "value": "LightGBM"},
                                             {"label": html.Span("Ridge"), "value": "Ridge Regression"},
                                             {"label": html.Span("Naïve (baseline)"), "value": "Naive"}],
                                    labelClassName="")], className="control"),
            html.Div([html.Span("Horizonte objetivo", className="control-label"),
                      slider("fc-h", min=1, max=30, step=1, value=7,
                                 marks={1: "t+1", 7: "7", 14: "14", 21: "21", 30: "30 días"},
                                 tooltip={"placement": "bottom", "always_visible": False})], className="control grow"),
            html.Div([html.Span("Histórico visible", className="control-label"),
                      seg("fc-win", [{"label": "60 d", "value": 60}, {"label": "6 m", "value": 180},
                                     {"label": "1 año", "value": 365}, {"label": "Todo", "value": 0}], 180)],
                     className="control"),
        ], className="controls"),
        html.Div([
            card([graph(gid="fc-chart")], title="Precio MIBGAS · proyección recursiva a 30 días",
                 sub=f"Último dato real: {fdate(H['last_date'])} · predicción hasta {fdate(H['forecast_end'])}",
                 foot="Cada predicción se reincorpora al histórico para calcular retardos y medias móviles del día siguiente."),
            card(html.Div(id="fc-tiles", className="fc-tiles"), title="Predicción en el horizonte",
                 sub="Diferencia frente al último precio real"),
        ], className="grid g-8-4"),

        html.Div("Validación", className="section-title"),
        html.Div([
            card([graph(C.backtest_rmse(theme))], title="Degradación del error con el horizonte",
                 sub="Backtesting recursivo desde 5 orígenes del conjunto de test · RMSE por día de horizonte"),
            card([html.Div(id="fc-bt-table")], title="Error por horizonte y modelo",
                 sub="RMSE y R² a 1, 7, 14 y 30 días · Tabla 7 de la memoria (recalculada)"),
        ], className="grid g-7-5"),
        html.Div([
            card([
                html.Div([html.Span("Origen del backtest", className="control-label"),
                          seg("fc-origin", [{"label": fdate(p["origin"]), "value": i}
                                            for i, p in enumerate(D.backtest()[1])], 0)],
                         style={"display": "flex", "flexDirection": "column", "gap": "7px", "marginBottom": "4px"}),
                graph(gid="fc-paths"),
            ], title="Trayectorias recursivas frente a la realidad", sub="30 días simulando un entorno productivo"),
            card([
                html.Div([html.Span("Modelo", className="control-label"),
                          seg("fc-test-model", [{"label": "LightGBM", "value": "LightGBM"},
                                                {"label": "Ridge", "value": "Ridge Regression"}], "LightGBM")],
                         style={"display": "flex", "flexDirection": "column", "gap": "7px", "marginBottom": "4px"}),
                graph(gid="fc-test"),
            ], title="Predicción a un paso sobre el test", sub="20 % final de la serie · real frente a predicho (t+1)"),
        ], className="grid g-2"),
        html.Div([
            card(holdout_table(), title="Hold-out a un paso (t+1)",
                 sub="Métricas sobre el 20 % final · €/MWh"),
            card([
                html.Div([
                    html.Div([html.Span("Exposición mensual (MWh de gas)", className="control-label"),
                              dcc.Input(id="fc-mwh", type="number", value=500, min=0, step=50, className="num-input",
                                        debounce=True)], className="control"),
                    html.Div(id="fc-cost-note", className="t2", style={"fontSize": "13px", "maxWidth": "420px"}),
                ], style={"display": "flex", "gap": "24px", "alignItems": "flex-end", "flexWrap": "wrap",
                          "marginBottom": "6px"}),
                graph(gid="fc-cost"),
            ], title="Simulador de impacto logístico",
                sub="Coste de aprovisionamiento en los próximos 30 días según cada proyección"),
        ], className="grid g-5-7"),
    ])


def holdout_table():
    hm = D.gas_holdout_metrics()
    tp = D.gas_test_predictions()
    import numpy as np
    err = tp["Naive"] - tp["Real"]
    naive = {"Model": "Naïve (t-1)", "RMSE": float(np.sqrt((err ** 2).mean())), "MAE": float(err.abs().mean()),
             "R2": float(1 - (err ** 2).sum() / ((tp["Real"] - tp["Real"].mean()) ** 2).sum())}
    rows_data = hm.to_dict("records") + [naive]
    rows_data = sorted(rows_data, key=lambda r: r["RMSE"])
    rows = []
    for r in rows_data:
        name = r["Model"]
        cls = "best" if r is rows_data[0] else ""
        sw = {"LightGBM": "bg-blue", "Ridge Regression": "bg-orange", "Naïve (t-1)": "bg-aqua"}.get(name)
        rows.append(html.Tr([
            html.Td([html.Span(className=f"swatch {sw}") if sw else html.Span(className="swatch bg-muted"), name],
                    className="name"),
            html.Td(fnum(r["RMSE"])), html.Td(fnum(r["MAE"])), html.Td(fnum(r["R2"], 3)),
        ], className=cls))
    return html.Div([
        html.Table([html.Thead(html.Tr([html.Th("Modelo"), html.Th("RMSE"), html.Th("MAE"), html.Th("R²")])),
                    html.Tbody(rows)], className="tbl"),
        html.Div(style={"height": "12px"}),
        insight([html.B("Paseo aleatorio. "), "Incluso a un día vista, repetir el precio de ayer es un rival muy "
                 "exigente: comportamiento típico de mercados eficientes (Hyndman & Athanasopoulos, 2021)."]),
    ])


def page_shap(theme):
    return html.Div([
        html.Div([
            html.Div([html.Span("Modelo explicado", className="control-label"),
                      seg("sh-model", [{"label": "LightGBM · TreeSHAP", "value": "LightGBM"},
                                       {"label": "Ridge · SHAP lineal", "value": "Ridge Regression"}], "LightGBM")],
                     className="control"),
            html.Div(id="sh-desc", className="t2", style={"fontSize": "13px", "maxWidth": "640px"}),
        ], className="controls"),
        html.Div([
            card([graph(gid="sh-bar")], title="Importancia global", sub="|SHAP| medio sobre el conjunto de test · % del total"),
            card([graph(gid="sh-bee")], title="Resumen SHAP (beeswarm)",
                 sub="Cada punto es un día · color: valor de la variable (azul bajo, rojo alto)"),
        ], className="grid g-5-7"),
        html.Div([
            card([graph(gid="sh-water")], title="¿Por qué predice lo que predice mañana?",
                 sub=f"Descomposición SHAP de la predicción para {fdate(D.future_forecast()['Trading_Day'].iloc[0])}"),
            card([
                insight([html.B("Memoria a corto plazo. "), "Los retardos de 1 y 7 días dominan ambos modelos: "
                         "el pasado pierde relevancia a medida que se aleja."]),
                html.Div(style={"height": "12px"}),
                insight([html.B("Geopolítica, casi irrelevante a diario. "), "Año, mes y variables de conflicto apenas "
                         "mueven la predicción: su efecto ya está incorporado en el precio de ayer."], gas=True),
                html.Div(style={"height": "12px"}),
                insight([html.B("Electricidad (Random Forest). "), "En el problema semestral, Año y la identidad del país "
                         "dominan: Alemania, Bélgica, Irlanda o Dinamarca empujan el precio al alza; Bosnia, Georgia o "
                         "Turquía, a la baja."]),
                html.Div(style={"height": "12px"}),
                html.Div("SHAP se calcula en el propio dashboard: TreeSHAP exacto nativo de LightGBM "
                         "(pred_contrib) y la forma cerrada lineal para Ridge.", className="muted",
                         style={"fontSize": "12px"}),
            ], title="Claves de interpretación"),
        ], className="grid g-7-5"),
    ])


def page_method(theme):
    steps = [
        ("01", "Adquisición", "Eurostat (nrg_pc_204) en Excel multi-hoja y MIBGAS en CSV anuales 2021–2026."),
        ("02", "Limpieza", "Símbolos estadísticos, nulos (Reino Unido), negativos y tipos; normalización de texto."),
        ("03", "Integración", "Gas diario agregado a semestres y convertido a €/kWh para unirlo con Eurostat."),
        ("04", "Variables", "Calendario cíclico, retardos, ventanas móviles y dummies geopolíticas sin fuga de información."),
        ("05", "Modelado", "8 algoritmos con partición cronológica y escalado ajustado sólo en entrenamiento."),
        ("06", "Evaluación", "MSE, RMSE, MAE, R², backtesting recursivo a 30 días e interpretabilidad SHAP."),
    ]
    feats = {
        "Temporales": ["Year", "Month", "Day", "Day_of_Week", "Day_of_Year", "Quarter", "Semester_Number",
                       "Month_Sin", "Month_Cos", "Day_of_Week_Sin", "Day_of_Week_Cos"],
        "Retardos": ["Gas_Price_Lag_1", "Gas_Price_Lag_7", "Gas_Price_Lag_14", "Gas_Price_Lag_30",
                     "Gas_Price_Pct_Change_Lag_1", "Gas_Trend"],
        "Ventanas móviles": ["Gas_Rolling_Mean_7", "Gas_Rolling_Mean_30", "Gas_Rolling_Std_7"],
        "Geopolíticas": ["Ukraine_War", "Energy_Crisis", "Middle_East_Conflict"],
    }
    models = D.gas_models()
    src = ("results/models/*.pkl" if all(v["source"] == "pkl" for v in models.values())
           else ", ".join(f"{k}: {v['source']}" for k, v in models.items()))
    return html.Div([
        html.Div("Pipeline", className="section-title"),
        html.Div([html.Div([html.Div(n, className="step-num"), html.Div(t, className="step-title"),
                            html.Div(x, className="step-text")], className="step") for n, t, x in steps],
                 className="pipeline"),
        html.Div("Conjuntos de datos", className="section-title"),
        html.Div([
            card([html.Dl([
                html.Dt("Observaciones"), html.Dd(f"{fnum(H['n_obs_eu'], 0)} filas × 26 variables"),
                html.Dt("Periodo"), html.Dd("2021-S1 → 2025-S2 (10 semestres)"),
                html.Dt("Cobertura"), html.Dd(f"{H['n_countries']} países europeos"),
                html.Dt("Banda"), html.Dd("Doméstica DC · 2.500–4.999 kWh/año"),
                html.Dt("Objetivo"), html.Dd("Electricity_Price (€/kWh)"),
                html.Dt("Validación"), html.Dd("Último año como test (cronológica)"),
            ], className="spec")], title="Eurostat + MIBGAS (semestral)", sub="Análisis gas–electricidad y modelos eléctricos"),
            card([html.Dl([
                html.Dt("Observaciones"), html.Dd(f"{fnum(H['n_days'], 0)} días × 26 variables"),
                html.Dt("Periodo"), html.Dd(f"{fdate(H['gas_start'])} → {fdate(H['gas_end'])}"),
                html.Dt("Producto"), html.Dd("GDAES D+1 · PVB España"),
                html.Dt("Objetivo"), html.Dd("Gas_Price (€/MWh)"),
                html.Dt("Validación"), html.Dd("80 / 20 cronológico + backtesting recursivo"),
                html.Dt("Modelos cargados"), html.Dd(src),
            ], className="spec")], title="MIBGAS diario", sub="Problema predictivo principal"),
        ], className="grid g-2"),
        html.Div("Ingeniería de características (modelo del gas)", className="section-title"),
        html.Div([card([html.Div([html.Span(f, className="feat") for f in fl], className="feat-list")],
                       title=g, sub=f"{len(fl)} variables") for g, fl in feats.items()], className="grid g-4"),
        html.Div([
            card(html.Ul([
                html.Li("Datos eléctricos sólo semestrales: pocos puntos para modelos complejos."),
                html.Li("MIBGAS representa el mercado ibérico; el resto de Europa tiene hubs propios (TTF…)."),
                html.Li("Los modelos no anticipan eventos sin precedentes: requieren monitorización y reentrenamiento."),
                html.Li("La predicción recursiva acumula error más allá de ~7 días."),
            ], className="bullets"), title="Limitaciones"),
            card(html.Ul([
                html.Li("Incorporar predicciones meteorológicas para anticipar renovables y picos de demanda."),
                html.Li("Pasar a datos horarios del mercado mayorista (curva diaria, almacenamiento)."),
                html.Li("Evaluar LSTM y Transformers si aumenta el volumen de datos."),
                html.Li("Combinar LightGBM (prudente) y Ridge (tendencia) en un rango de confianza."),
            ], className="bullets"), title="Trabajo futuro"),
        ], className="grid g-2"),
    ])


RENDER = {"/": page_home, "/mercado": page_market, "/relacion": page_relation, "/modelos": page_models,
          "/prediccion": page_forecast, "/shap": page_shap, "/metodologia": page_method}


# ==================================================================
# CALLBACKS
# ==================================================================
@app.callback(Output("page", "children"), Output("page-title", "children"), Output("page-eyebrow", "children"),
              *[Output(f"nav-{p.strip('/') or 'home'}", "className") for p in RENDER],
              Input("url", "pathname"), Input("theme", "data"))
def route(path, theme):
    path = path if path in RENDER else "/"
    name, desc = PAGE_META[path]
    classes = ["nav-link active" if p == path else "nav-link" for p in RENDER]
    return (RENDER[path](theme), name, desc, *classes)


@app.callback(Output("theme", "data"), Input("theme-btn", "n_clicks"), State("theme", "data"),
              prevent_initial_call=True)
def toggle_theme(_, theme):
    return "light" if theme == "dark" else "dark"


app.clientside_callback(
    """function(theme){
        document.documentElement.setAttribute('data-theme', theme || 'dark');
        return theme === 'light' ? 'Modo oscuro' : 'Modo claro';
    }""",
    Output("theme-label", "children"), Input("theme", "data"))


# ---------------- Mercado europeo ----------------
@app.callback(Output("mk-europe", "figure"), Output("mk-iberia", "figure"), Output("mk-box", "figure"),
              Output("mk-hist-e", "figure"), Output("mk-kpis", "children"),
              Input("mk-tax", "value"), Input("mk-countries", "value"), Input("theme", "data"))
def update_market(tax, countries, theme):
    countries = (countries or ["Spain"])[:4]
    rank = D.country_ranking(tax)
    es = rank[rank["Country"] == "Spain"].iloc[0]
    eu = D.europe_mean_by_semester(tax)
    peak = D.country_by_semester(tax, ["Spain"]).sort_values("Price").iloc[-1]
    f_e, _ = C.distributions(theme, tax)
    kpis = [
        kpi("Precio medio España", fnum(es["Price"], 3), "€/kWh", f"Puesto {int(es['Rank'])} de {len(rank)} países", "blue"),
        kpi("Media europea", fnum(rank["Price"].mean(), 3), "€/kWh",
            [html.Span(pct(es["Price"] / rank["Price"].mean() - 1), className="delta up"), " España sobre la media"]),
        kpi("Pico de España", fnum(peak["Price"], 3), "€/kWh", f"Semestre {peak['Semester']}", "blue"),
        kpi("País más caro", rank.iloc[0]["Country_ES"], "", text=True, note=
            f"{fnum(rank.iloc[0]['Price'], 3)} €/kWh · más barato: {rank.iloc[-1]['Country_ES']} "
            f"({fnum(rank.iloc[-1]['Price'], 3)})"),
    ]
    return (C.spain_vs_europe(theme, tax, countries), C.iberia_vs_gas(theme, tax),
            C.top15_boxplot(theme, tax), f_e, kpis)


@app.callback(Output("mk-map", "figure"), Output("mk-rank", "figure"),
              Input("mk-tax", "value"), Input("mk-sem", "value"), Input("theme", "data"))
def update_map(tax, sem_idx, theme):
    sem = "all" if not sem_idx else SEMS[sem_idx - 1]
    return C.europe_map(theme, tax, sem), C.country_ranking_bar(theme, tax, sem)


# ---------------- Relación gas-electricidad ----------------
@app.callback(Output("rl-heat", "figure"), Output("rl-scatter", "figure"), Output("rl-kpis", "children"),
              Output("rl-table", "children"), Input("rl-tax", "value"), Input("theme", "data"))
def update_relation(tax, theme):
    corr = D.spearman_iberia(tax)
    kpis = [
        kpi("Gas ↔ electricidad", fnum(corr.loc["Electricidad", "Precio gas"]), "ρ",
            "Correlación de rangos de Spearman", "blue"),
        kpi("Guerra de Ucrania ↔ electricidad", fnum(corr.loc["Electricidad", "Guerra Ucrania"]), "ρ",
            "Variable binaria desde 24/02/2022", "orange"),
        kpi("Crisis energética ↔ electricidad", fnum(corr.loc["Electricidad", "Crisis energética"]), "ρ",
            "Sep 2021 – mar 2023", "orange"),
    ]
    ss = D.semester_summary()
    rows = [html.Tr([html.Td(r["Semester"], className="name"),
                     html.Td(fnum(r["Avg_Electricity_Price"], 4)), html.Td(fnum(r["Median_Electricity_Price"], 4)),
                     html.Td(fnum(r["Std_Electricity_Price"], 4)),
                     html.Td(fnum(r["Gas_Price"], 4)), html.Td(fnum(r["Gas_Price_Median"], 4)),
                     html.Td(fnum(r["Gas_Price_Std"], 4))]) for _, r in ss.iterrows()]
    table = html.Table([
        html.Thead([
            html.Tr([html.Th(""), html.Th(html.Span([html.Span(className="swatch bg-blue"), "Electricidad"]), colSpan=3,
                                          style={"textAlign": "center"}),
                     html.Th(html.Span([html.Span(className="swatch bg-orange"), "Gas"]), colSpan=3,
                             style={"textAlign": "center"})]),
            html.Tr([html.Th("Semestre"), html.Th("Media"), html.Th("Mediana"), html.Th("Desv."),
                     html.Th("Media"), html.Th("Mediana"), html.Th("Desv.")])]),
        html.Tbody(rows)], className="tbl")
    return C.spearman_heatmap(theme, tax), C.iberia_scatter(theme, tax), kpis, table


# ---------------- Predicción ----------------
@app.callback(Output("fc-chart", "figure"), Output("fc-tiles", "children"),
              Input("fc-models", "value"), Input("fc-h", "value"), Input("fc-win", "value"), Input("theme", "data"))
def update_forecast(models, h, win, theme):
    models = models or []
    ff = D.future_forecast()
    target_date = ff["Trading_Day"].iloc[h - 1]
    last = H["last_price"]
    bt, _ = D.backtest()
    tiles = [html.Div([
        html.Div("Precio real de referencia", className="kpi-label"),
        html.Div([fnum(last), html.Span("€/MWh", className="kpi-unit")], className="kpi-value"),
        html.Div(fdate(H["last_date"]), className="kpi-note"),
    ], className="fc-tile")]
    for m in ["LightGBM", "Ridge Regression", "Naive"]:
        v = ff[m].iloc[h - 1]
        d = v / last - 1
        rmse = bt[(bt["Model"] == m) & (bt["Horizon"] == h)]["RMSE"].iloc[0]
        color = {"LightGBM": "blue", "Ridge Regression": "orange", "Naive": "aqua"}[m]
        tiles.append(html.Div([
            html.Span(className=f"kpi-accent bg-{color}"),
            html.Div([html.Div(f"{MODEL_LABEL[m]} · t+{h}", className="kpi-label"),
                      html.Span(pct(d), className=f"delta {'up' if d > 0.0005 else 'down' if d < -0.0005 else 'neutral'}")],
                     className="fc-row"),
            html.Div([fnum(v), html.Span("€/MWh", className="kpi-unit")], className="kpi-value"),
            html.Div(f"{fdate(target_date)} · error típico ±{fnum(rmse, 1)} €/MWh", className="kpi-note"),
        ], className="fc-tile", style={"opacity": 1 if m in models else 0.45}))
    return C.forecast_chart(theme, models, h, win), tiles


@app.callback(Output("fc-bt-table", "children"), Input("theme", "data"))
def bt_table(_):
    bt, _ = D.backtest()
    hs = [1, 7, 14, 30]
    rows = []
    for m in ["Naive", "LightGBM", "Ridge Regression"]:
        d = bt[bt["Model"] == m].set_index("Horizon")
        sw = {"LightGBM": "bg-blue", "Ridge Regression": "bg-orange", "Naive": "bg-aqua"}[m]
        cells = []
        for h in hs:
            best = bt[bt["Horizon"] == h]["RMSE"].min()
            v = d.loc[h, "RMSE"]
            cells.append(html.Td(html.Span(fnum(v), style={"fontWeight": 650, "color": "var(--text)"} if v == best else {})))
        rows.append(html.Tr([html.Td([html.Span(className=f"swatch {sw}"), MODEL_LABEL[m]], className="name")] + cells))
    r2rows = []
    for m in ["Naive", "LightGBM", "Ridge Regression"]:
        d = bt[bt["Model"] == m].set_index("Horizon")
        r2rows.append(html.Tr([html.Td(MODEL_LABEL[m], className="name")] +
                              [html.Td(fnum(d.loc[h, "R2"], 2)) for h in hs]))
    head = html.Thead(html.Tr([html.Th("RMSE (€/MWh)")] + [html.Th(f"{h} d") for h in hs]))
    head2 = html.Thead(html.Tr([html.Th("R²")] + [html.Th(f"{h} d") for h in hs]))
    return html.Div([
        html.Table([head, html.Tbody(rows)], className="tbl"),
        html.Div(style={"height": "14px"}),
        html.Table([head2, html.Tbody(r2rows)], className="tbl"),
        html.Div(style={"height": "14px"}),
        insight([html.B("Conclusión: "), "LightGBM responde bien a 1 día, pero a 30 días el Naïve acota el error "
                 "(RMSE ≈ 2,6 €/MWh). Ridge colapsa (R² < 0) al proyectar rígidamente la última pendiente."], gas=True),
    ])


@app.callback(Output("fc-paths", "figure"), Input("fc-origin", "value"), Input("theme", "data"))
def update_paths(i, theme):
    return C.backtest_paths(theme, i or 0)


@app.callback(Output("fc-test", "figure"), Input("fc-test-model", "value"), Input("theme", "data"))
def update_test(m, theme):
    return C.test_actual_vs_pred(theme, m)


@app.callback(Output("fc-cost", "figure"), Output("fc-cost-note", "children"),
              Input("fc-mwh", "value"), Input("fc-models", "value"), Input("theme", "data"))
def update_cost(mwh, models, theme):
    mwh = float(mwh or 0)
    ff = D.future_forecast()
    daily = mwh / 30
    models = models or ["Naive"]
    costs = {m: float(ff[m].sum() * daily) for m in ["Naive", "LightGBM", "Ridge Regression"] if m in models}
    if not costs:
        return C.empty_fig(theme, "Selecciona al menos un modelo"), ""
    lo, hi = min(costs.values()), max(costs.values())
    note = [f"Consumo diario ≈ {fnum(daily, 1)} MWh. Horquilla entre proyecciones: ",
            html.B(f"{fnum(hi - lo, 0)} €"), f" ({fnum((hi / lo - 1) * 100, 1)} %). Útil para decidir si adelantar "
            "compras o cubrir riesgo (hedging)."]
    return C.cost_bars(theme, costs), note


# ---------------- SHAP ----------------
@app.callback(Output("sh-bar", "figure"), Output("sh-bee", "figure"), Output("sh-water", "figure"),
              Output("sh-desc", "children"), Input("sh-model", "value"), Input("theme", "data"))
def update_shap(model, theme):
    imp = D.shap_importance(model)
    top = imp.iloc[0]
    desc = [f"En {MODEL_LABEL[model]}, ", html.B(top["Label"]),
            f" explica el {fnum(top['Share'] * 100, 0)} % del impacto total; las tres primeras variables suman el "
            f"{fnum(imp['Share'].head(3).sum() * 100, 0)} %."]
    return C.shap_bar(theme, model), C.shap_beeswarm(theme, model), C.shap_waterfall(theme, model), desc


if __name__ == "__main__":
    import os
    app.run(debug=False, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8050)))
