"""Constructores de figuras Plotly con el sistema visual de VOLT."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import data as D
from theme import FONT, MODEL_COLOR, MODEL_LABEL, T


# ------------------------------------------------------------------
# Base común
# ------------------------------------------------------------------
def base_layout(fig: go.Figure, theme: str, height: int = 360, legend: bool = True,
                margin=None, hovermode="x unified") -> go.Figure:
    c = T(theme)
    fig.update_layout(
        height=height,
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=c["text2"]),
        margin=margin or dict(l=8, r=16, t=12, b=8),
        separators=",.",
        hovermode=hovermode,
        hoverlabel=dict(bgcolor=c["hover_bg"], bordercolor=c["axis"],
                        font=dict(family=FONT, size=12, color=c["text"])),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=12, color=c["text2"]), bgcolor="rgba(0,0,0,0)",
                    itemclick="toggle", itemdoubleclick="toggleothers"),
    )
    axis = dict(showgrid=True, gridcolor=c["grid"], gridwidth=1, zeroline=False,
                linecolor=c["axis"], showline=False, ticks="", automargin=True,
                tickfont=dict(size=11, color=c["muted"]),
                title=dict(font=dict(size=11, color=c["muted"])))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis, ticklabelstandoff=6)
    fig.update_xaxes(showgrid=False, showline=True)
    return fig


def empty_fig(theme: str, msg: str, height=300) -> go.Figure:
    fig = go.Figure()
    base_layout(fig, theme, height, legend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text=msg, showarrow=False, font=dict(color=T(theme)["muted"], size=13))
    return fig


def _event_bands(fig: go.Figure, theme: str, x_min=None, x_max=None, labels=True):
    c = T(theme)
    x_min = pd.Timestamp(x_min) if x_min is not None else None
    x_max = pd.Timestamp(x_max) if x_max is not None else None
    for i, ev in enumerate(D.EVENTS):
        start = pd.Timestamp(ev["start"])
        end = pd.Timestamp(ev["end"]) if ev["end"] else None
        if x_max is not None and start > x_max:
            continue
        if end is not None:
            x0 = max(start, x_min) if x_min is not None else start
            x1 = min(end, x_max) if x_max is not None else end
            if x1 <= x0:
                continue
            fig.add_vrect(x0=x0, x1=x1, fillcolor=c["band"], line_width=0, layer="below")
            anchor = x0
        else:
            if x_min is not None and start < x_min:
                continue
            fig.add_vline(x=start, line_width=1, line_color=c["axis"])
            anchor = start
        if labels:
            fig.add_annotation(x=anchor, xref="x", y=[1.0, 0.93, 0.86, 1.0][i % 4], yref="paper",
                               text=ev["label"], showarrow=False, xanchor="left", xshift=6,
                               font=dict(size=10.5, color=c["band_label"]))


# ------------------------------------------------------------------
# RESUMEN
# ------------------------------------------------------------------
def gas_history(theme: str, height=380) -> go.Figure:
    c = T(theme)
    g = D.load_gas()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["Trading_Day"], y=g["Gas_Price"], mode="lines", name="MIBGAS diario",
        line=dict(color=c["orange"], width=1.6), fill="tozeroy",
        fillcolor="rgba(217,89,38,0.10)" if theme != "light" else "rgba(235,104,52,0.10)",
        hovertemplate="%{y:.2f} €/MWh<extra></extra>"))
    _event_bands(fig, theme)
    h = D.headline()
    fig.add_annotation(x=h["peak_date"], y=h["peak_price"], text=f"Máximo {h['peak_price']:.0f} €/MWh",
                       showarrow=True, arrowhead=0, arrowcolor=c["muted"], ax=64, ay=0,
                       font=dict(size=11, color=c["text"]), xanchor="left")
    fig.add_trace(go.Scatter(x=[h["last_date"]], y=[h["last_price"]], mode="markers",
                             marker=dict(size=9, color=c["orange"], line=dict(color=c["surface"], width=2)),
                             hoverinfo="skip", showlegend=False))
    base_layout(fig, theme, height, legend=False, margin=dict(l=8, r=24, t=30, b=8))
    fig.update_yaxes(title_text="€/MWh", rangemode="tozero")
    fig.update_xaxes(rangeslider=dict(visible=False))
    return fig


# ------------------------------------------------------------------
# MERCADO EUROPEO
# ------------------------------------------------------------------
def _sem_index(sem: str) -> int:
    return D.semesters().index(sem)


def _iberian_band(fig, theme, label=True):
    c = T(theme)
    x0, x1 = _sem_index("2022-S1") + 0.5, _sem_index("2023-S2") + 0.5
    fig.add_vrect(x0=x0, x1=x1, fillcolor=c["band"], line_width=0, layer="below")
    if label:
        fig.add_annotation(x=x0, y=1, yref="paper", text="Excepción Ibérica", showarrow=False,
                           xanchor="left", xshift=6, font=dict(size=10.5, color=c["band_label"]))


COUNTRY_SLOTS = ["blue", "aqua", "yellow", "violet"]


def spain_vs_europe(theme: str, tax: str, countries: list[str], height=360) -> go.Figure:
    c = T(theme)
    sems = D.semesters()
    fig = go.Figure()
    eu = D.europe_mean_by_semester(tax)
    fig.add_trace(go.Scatter(x=eu["Semester"], y=eu["Price"], name="Media europea (40 países)",
                             mode="lines", line=dict(color=c["neutral"], width=2, dash="dot"),
                             hovertemplate="%{y:.3f} €/kWh"))
    fig.add_annotation(x=eu["Semester"].iloc[-1], y=eu["Price"].iloc[-1], text="Media UE", showarrow=False,
                       xanchor="left", xshift=8, font=dict(size=11, color=c["muted"]))
    cs = D.country_by_semester(tax, countries)
    for i, country in enumerate(countries):
        d = cs[cs["Country"] == country]
        if d.empty:
            continue
        col = c[COUNTRY_SLOTS[i % len(COUNTRY_SLOTS)]]
        name = D.COUNTRY_ES.get(country, country)
        fig.add_trace(go.Scatter(x=d["Semester"], y=d["Price"], name=name, mode="lines+markers",
                                 line=dict(color=col, width=2.2), marker=dict(size=7, color=col,
                                 line=dict(color=c["surface"], width=1.5)),
                                 hovertemplate="%{y:.3f} €/kWh"))
        fig.add_annotation(x=d["Semester"].iloc[-1], y=d["Price"].iloc[-1], text=name, showarrow=False,
                           xanchor="left", xshift=8, font=dict(size=11, color=c["text2"]))
    _iberian_band(fig, theme)
    base_layout(fig, theme, height, margin=dict(l=8, r=70, t=36, b=8))
    fig.update_xaxes(categoryorder="array", categoryarray=sems)
    fig.update_yaxes(title_text="€/kWh", tickformat=".2f")
    return fig


def iberia_vs_gas(theme: str, tax: str, height=360) -> go.Figure:
    """Electricidad ES/PT y gas ibérico — misma unidad (€/kWh), un solo eje."""
    c = T(theme)
    sems = D.semesters()
    cs = D.country_by_semester(tax, ["Spain", "Portugal"])
    gas = D.gas_by_semester()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gas["Semester"], y=gas["Gas"], name="Gas MIBGAS",
                             mode="lines+markers", line=dict(color=c["orange"], width=2.2),
                             marker=dict(size=7, line=dict(color=c["surface"], width=1.5)),
                             fill="tozeroy",
                             fillcolor="rgba(217,89,38,0.08)",
                             hovertemplate="%{y:.3f} €/kWh"))
    for country, slot in (("Spain", "blue"), ("Portugal", "aqua")):
        d = cs[cs["Country"] == country]
        fig.add_trace(go.Scatter(x=d["Semester"], y=d["Price"], name=D.COUNTRY_ES[country],
                                 mode="lines+markers", line=dict(color=c[slot], width=2.2),
                                 marker=dict(size=7, line=dict(color=c["surface"], width=1.5)),
                                 hovertemplate="%{y:.3f} €/kWh"))
        fig.add_annotation(x=d["Semester"].iloc[-1], y=d["Price"].iloc[-1], text=D.COUNTRY_ES[country],
                           showarrow=False, xanchor="left", xshift=8, font=dict(size=11, color=c["text2"]))
    fig.add_annotation(x=gas["Semester"].iloc[-1], y=gas["Gas"].iloc[-1], text="Gas", showarrow=False,
                       xanchor="left", xshift=8, font=dict(size=11, color=c["text2"]))
    _iberian_band(fig, theme)
    base_layout(fig, theme, height, margin=dict(l=8, r=70, t=36, b=8))
    fig.update_xaxes(categoryorder="array", categoryarray=sems)
    fig.update_yaxes(title_text="€/kWh", tickformat=".2f", rangemode="tozero")
    return fig


def europe_map(theme: str, tax: str, semester: str, height=560) -> go.Figure:
    c = T(theme)
    r = D.country_ranking(tax, semester)
    r = r[r["ISO3"].notna() & (r["ISO3"] != "XKX")]
    scale = [[i / (len(c["seq"]) - 1), col] for i, col in enumerate(c["seq"])]
    fig = go.Figure(go.Choropleth(
        locations=r["ISO3"], z=r["Price"], text=r["Country_ES"],
        colorscale=scale, zmin=0.0, zmax=0.40,
        marker_line_color=c["surface"], marker_line_width=0.8,
        colorbar=dict(title=dict(text="€/kWh", font=dict(size=11, color=c["muted"])),
                      thickness=10, len=0.6, x=0.98, outlinewidth=0,
                      tickfont=dict(size=10, color=c["muted"]), tickformat=".2f"),
        hovertemplate="<b>%{text}</b><br>%{z:.3f} €/kWh<extra></extra>"))
    fig.update_geos(scope="europe", resolution=50, projection_type="mercator", showframe=False,
                    showcoastlines=False, showland=True, landcolor=c["surface2"],
                    showcountries=True, countrycolor=c["surface"], bgcolor="rgba(0,0,0,0)",
                    lataxis_range=[34.5, 71], lonaxis_range=[-24, 44], showocean=False, showlakes=False)
    base_layout(fig, theme, height, legend=False, margin=dict(l=0, r=0, t=0, b=0), hovermode="closest")
    return fig


def country_ranking_bar(theme: str, tax: str, semester: str, height=640) -> go.Figure:
    c = T(theme)
    r = D.country_ranking(tax, semester).sort_values("Price")
    highlight = {"Spain": c["blue"], "Portugal": c["aqua"]}
    colors = [highlight.get(x, c["neutral_bar"]) for x in r["Country"]]
    eu_mean = r["Price"].mean()
    fig = go.Figure(go.Bar(
        x=r["Price"], y=r["Country_ES"], orientation="h", marker=dict(color=colors, cornerradius=3),
        customdata=r["Rank"],
        hovertemplate="<b>%{y}</b><br>%{x:.3f} €/kWh · puesto %{customdata}<extra></extra>"))
    fig.add_vline(x=eu_mean, line_width=1, line_dash="dot", line_color=c["text2"])
    fig.add_annotation(x=eu_mean, y=1.0, yref="paper", text=f"Media {eu_mean:.3f}".replace(".", ","),
                       showarrow=False, xanchor="left", xshift=4, yanchor="bottom",
                       font=dict(size=10.5, color=c["text2"]))
    base_layout(fig, theme, height, legend=False, margin=dict(l=8, r=16, t=22, b=8), hovermode="closest")
    fig.update_layout(bargap=0.28)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=c["text2"]))
    fig.update_xaxes(showgrid=True, title_text="€/kWh", tickformat=".2f")
    return fig


def top15_boxplot(theme: str, tax: str, height=380) -> go.Figure:
    c = T(theme)
    top = D.country_ranking(tax).head(15)["Country"].tolist()
    df = D.country_prices(tax, top)
    fig = go.Figure()
    for country in top:
        d = df[df["Country"] == country]
        is_es = country == "Spain"
        col = c["blue"] if is_es else c["neutral"]
        fig.add_trace(go.Box(y=d["Price"], name=D.COUNTRY_ES[country], marker=dict(color=col, size=4),
                             line=dict(color=col, width=1.4),
                             fillcolor="rgba(57,135,229,0.25)" if is_es else c["neutral_soft"],
                             boxpoints="outliers", hovertemplate="%{y:.3f} €/kWh<extra>" + D.COUNTRY_ES[country] + "</extra>"))
    base_layout(fig, theme, height, legend=False, hovermode="closest")
    fig.update_yaxes(title_text="€/kWh", tickformat=".2f")
    fig.update_xaxes(tickangle=-35)
    return fig


def distributions(theme: str, tax: str, height=300) -> tuple[go.Figure, go.Figure]:
    c = T(theme)
    df = D.filter_tax(tax)
    f1 = go.Figure(go.Histogram(x=df["Price"], nbinsx=40, marker=dict(color=c["blue"], cornerradius=2),
                                hovertemplate="%{x} €/kWh<br>%{y} obs.<extra></extra>"))
    base_layout(f1, theme, height, legend=False, hovermode="closest")
    f1.update_layout(bargap=0.08)
    f1.update_xaxes(title_text="Precio electricidad (€/kWh)", tickformat=".2f")
    f1.update_yaxes(title_text="Observaciones")
    g = D.load_gas()
    f2 = go.Figure(go.Histogram(x=g["Gas_Price"], nbinsx=50, marker=dict(color=c["orange"], cornerradius=2),
                                hovertemplate="%{x} €/MWh<br>%{y} días<extra></extra>"))
    base_layout(f2, theme, height, legend=False, hovermode="closest")
    f2.update_layout(bargap=0.08)
    f2.update_xaxes(title_text="Precio gas MIBGAS (€/MWh, diario)")
    f2.update_yaxes(title_text="Días")
    return f1, f2


# ------------------------------------------------------------------
# GAS ↔ ELECTRICIDAD
# ------------------------------------------------------------------
def spearman_heatmap(theme: str, tax: str, height=420) -> go.Figure:
    c = T(theme)
    corr = D.spearman_iberia(tax)
    z = corr.values
    labels = corr.columns.tolist()
    text = [[f"{v:.2f}".replace(".", ",") for v in row] for row in z]
    scale = [[0, c["div_neg"]], [0.5, c["div_mid"]], [1, c["div_pos"]]]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, zmin=-1, zmax=1, colorscale=scale, text=text,
        texttemplate="%{text}", textfont=dict(size=12, color=c["text"]), xgap=2, ygap=2,
        colorbar=dict(thickness=10, len=0.8, outlinewidth=0, tickfont=dict(size=10, color=c["muted"])),
        hovertemplate="%{y} · %{x}<br>ρ = %{z:.2f}<extra></extra>"))
    base_layout(fig, theme, height, legend=False, hovermode="closest", margin=dict(l=8, r=8, t=8, b=8))
    fig.update_xaxes(showgrid=False, showline=False, tickangle=-35, side="bottom")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


KEY_SEMS = {"2021-S1", "2021-S2", "2022-S1", "2022-S2", "2023-S1", "2025-S2"}


def iberia_scatter(theme: str, tax: str, height=420) -> go.Figure:
    c = T(theme)
    d = D.iberia_semester(tax)
    fig = go.Figure()
    # Recta de tendencia (MCO) sólo como guía visual
    k, b = np.polyfit(d["Gas_Price"], d["Price"], 1)
    xs = np.linspace(d["Gas_Price"].min(), d["Gas_Price"].max(), 20)
    fig.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines", line=dict(color=c["axis"], width=1.5, dash="dot"),
                             hoverinfo="skip", name="Tendencia lineal"))
    fig.add_trace(go.Scatter(
        x=d["Gas_Price"], y=d["Price"], mode="markers+text", text=d["Semester"],
        texttemplate=[t if t in KEY_SEMS else "" for t in d["Semester"]],
        textposition="top center", textfont=dict(size=10, color=c["muted"]),
        marker=dict(size=13, color=c["blue"], line=dict(color=c["surface"], width=2)),
        name="Semestre (media ES+PT)",
        hovertemplate="<b>%{text}</b><br>Gas %{x:.3f} €/kWh<br>Electricidad %{y:.3f} €/kWh<extra></extra>"))
    base_layout(fig, theme, height, legend=False, hovermode="closest")
    fig.update_xaxes(title_text="Precio del gas (€/kWh)", tickformat=".2f", showgrid=True)
    fig.update_yaxes(title_text="Precio de la electricidad (€/kWh)", tickformat=".2f")
    return fig


# ------------------------------------------------------------------
# MODELOS DE ELECTRICIDAD
# ------------------------------------------------------------------
def electricity_r2(theme: str, height=320) -> go.Figure:
    c = T(theme)
    m = D.electricity_model_results()
    m = m[m["Model"] != "Linear Regression"].sort_values("R2")
    best = m["R2"].max()
    colors = [c["blue"] if v == best else c["neutral"] for v in m["R2"]]
    fig = go.Figure(go.Bar(
        x=m["R2"], y=m["Model"], orientation="h", marker=dict(color=colors, cornerradius=3),
        text=[f"{v:.3f}".replace(".", ",") if v >= 0 else "" for v in m["R2"]], textposition="outside",
        textfont=dict(size=11, color=c["text2"]), cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>R² = %{x:.3f}<extra></extra>"))
    for _, r in m[m["R2"] < 0].iterrows():
        fig.add_annotation(x=0, y=r["Model"], text=f"{r['R2']:.3f}".replace(".", ","), showarrow=False,
                           xanchor="left", xshift=6, font=dict(size=11, color=c["critical"]))
    fig.add_vline(x=0, line_width=1, line_color=c["axis"])
    base_layout(fig, theme, height, legend=False, hovermode="closest", margin=dict(l=8, r=50, t=8, b=8))
    fig.update_layout(bargap=0.35)
    fig.update_xaxes(showgrid=True, title_text="R² en test (último año)", range=[-0.25, 1])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color=c["text2"]))
    return fig


GROUP_SLOT = {"País": "blue", "Mercado del gas": "orange", "Temporal": "aqua", "Geopolítica": "yellow"}


def rf_importance(theme: str, top_n=15, height=440) -> go.Figure:
    c = T(theme)
    fi = D.electricity_feature_importance().head(top_n).iloc[::-1]
    fig = go.Figure()
    for grp, slot in GROUP_SLOT.items():
        d = fi[fi["Group"] == grp]
        if d.empty:
            continue
        fig.add_trace(go.Bar(x=d["Importance"], y=d["Label"], orientation="h", name=grp,
                             marker=dict(color=c[slot], cornerradius=3),
                             hovertemplate="<b>%{y}</b><br>Importancia %{x:.3f}<extra></extra>"))
    base_layout(fig, theme, height, hovermode="closest")
    fig.update_layout(barmode="overlay", bargap=0.3)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=fi["Label"].tolist(),
                     tickfont=dict(size=11, color=c["text2"]))
    fig.update_xaxes(showgrid=True, title_text="Importancia (reducción de impureza)")
    return fig


def rf_group_share(theme: str, height=120) -> go.Figure:
    c = T(theme)
    fi = D.electricity_feature_importance()
    g = fi.groupby("Group")["Importance"].sum()
    g = g / g.sum()
    fig = go.Figure()
    for grp, slot in GROUP_SLOT.items():
        v = float(g.get(grp, 0))
        fig.add_trace(go.Bar(x=[v], y=[""], orientation="h", name=f"{grp} · {v*100:.0f} %",
                             marker=dict(color=c[slot], line=dict(color=c["surface"], width=2)),
                             hovertemplate=f"{grp}: {v*100:.1f} %<extra></extra>"))
    base_layout(fig, theme, height, hovermode="closest", margin=dict(l=4, r=4, t=4, b=4))
    fig.update_layout(barmode="stack", legend=dict(y=-0.25, yanchor="top"))
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False)
    return fig


# ------------------------------------------------------------------
# PREDICCIÓN DEL GAS
# ------------------------------------------------------------------
def forecast_chart(theme: str, models: list[str], horizon: int, window: int, height=440) -> go.Figure:
    c = T(theme)
    g = D.load_gas()
    ff = D.future_forecast()
    hist = g.tail(window) if window else g
    last_date, last_price = g["Trading_Day"].iloc[-1], g["Gas_Price"].iloc[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Trading_Day"], y=hist["Gas_Price"], name="Histórico MIBGAS",
                             mode="lines", line=dict(color=c["text2"], width=1.8),
                             fill="tozeroy", fillcolor=c["neutral_soft"],
                             hovertemplate="%{y:.2f} €/MWh"))
    fig.add_vrect(x0=last_date, x1=ff["Trading_Day"].iloc[-1], fillcolor=c["band"], line_width=0, layer="below")
    fig.add_vline(x=last_date, line_width=1, line_color=c["axis"])
    fig.add_annotation(x=last_date, y=1, yref="paper", text="Inicio de la predicción", showarrow=False,
                       xanchor="left", xshift=6, font=dict(size=10.5, color=c["band_label"]))
    target = ff["Trading_Day"].iloc[horizon - 1]
    for m in ["Naive", "LightGBM", "Ridge Regression"]:
        if m not in models:
            continue
        col = c[MODEL_COLOR[m]]
        xs = [last_date] + ff["Trading_Day"].tolist()
        ys = [last_price] + ff[m].tolist()
        fig.add_trace(go.Scatter(x=xs, y=ys, name=MODEL_LABEL[m], mode="lines",
                                 line=dict(color=col, width=2.4, dash="dash" if m == "Naive" else "solid"),
                                 hovertemplate="%{y:.2f} €/MWh"))
        yv = ff[m].iloc[horizon - 1]
        fig.add_trace(go.Scatter(x=[target], y=[yv], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=11, color=col, line=dict(color=c["surface"], width=2))))
        fig.add_annotation(x=ff["Trading_Day"].iloc[-1], y=ff[m].iloc[-1],
                           text=f"{MODEL_LABEL[m]} {ff[m].iloc[-1]:.1f}".replace(".", ","),
                           showarrow=False, xanchor="left", xshift=8, font=dict(size=11, color=c["text2"]))
    fig.add_vline(x=target, line_width=1, line_dash="dot", line_color=c["muted"])
    fig.add_trace(go.Scatter(x=[last_date], y=[last_price], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(size=9, color=c["text"], line=dict(color=c["surface"], width=2))))
    base_layout(fig, theme, height, margin=dict(l=8, r=96, t=36, b=8))
    fig.update_yaxes(title_text="€/MWh")
    fig.update_xaxes(range=[hist["Trading_Day"].iloc[0], ff["Trading_Day"].iloc[-1] + pd.Timedelta(days=2)])
    return fig


def backtest_rmse(theme: str, height=420) -> go.Figure:
    c = T(theme)
    bt, _ = D.backtest()
    fig = go.Figure()
    for m in ["Naive", "LightGBM", "Ridge Regression"]:
        d = bt[bt["Model"] == m].sort_values("Horizon")
        col = c[MODEL_COLOR[m]]
        sizes = [9 if h in (1, 7, 14, 30) else 0 for h in d["Horizon"]]
        fig.add_trace(go.Scatter(x=d["Horizon"], y=d["RMSE"], name=MODEL_LABEL[m], mode="lines+markers",
                                 line=dict(color=col, width=2.2, dash="dash" if m == "Naive" else "solid"),
                                 marker=dict(size=sizes, color=col, line=dict(color=c["surface"], width=1.5)),
                                 hovertemplate="%{y:.2f} €/MWh"))
        fig.add_annotation(x=30, y=d["RMSE"].iloc[-1], text=MODEL_LABEL[m], showarrow=False,
                           xanchor="left", xshift=8, font=dict(size=11, color=c["text2"]))
    base_layout(fig, theme, height, margin=dict(l=8, r=64, t=36, b=8))
    fig.update_xaxes(title_text="Horizonte de predicción (días)", tickvals=[1, 7, 14, 21, 30], range=[0.5, 30.5])
    fig.update_yaxes(title_text="RMSE (€/MWh)", rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    return fig


def backtest_paths(theme: str, origin_idx: int, height=340) -> go.Figure:
    c = T(theme)
    _, paths = D.backtest()
    p = paths[origin_idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p["dates"], y=p["actual"], name="Real", mode="lines",
                             line=dict(color=c["text2"], width=2.2), hovertemplate="%{y:.2f} €/MWh"))
    for m in ["Naive", "LightGBM", "Ridge Regression"]:
        fig.add_trace(go.Scatter(x=p["dates"], y=p[m], name=MODEL_LABEL[m], mode="lines",
                                 line=dict(color=c[MODEL_COLOR[m]], width=2, dash="dash" if m == "Naive" else "solid"),
                                 hovertemplate="%{y:.2f} €/MWh"))
    base_layout(fig, theme, height, margin=dict(l=8, r=16, t=36, b=8))
    fig.update_yaxes(title_text="€/MWh")
    return fig


def test_actual_vs_pred(theme: str, model: str, height=340) -> go.Figure:
    c = T(theme)
    tp = D.gas_test_predictions()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tp["Trading_Day"], y=tp["Real"], name="Real", mode="lines",
                             line=dict(color=c["text2"], width=1.6), hovertemplate="%{y:.2f} €/MWh"))
    fig.add_trace(go.Scatter(x=tp["Trading_Day"], y=tp[model], name=f"{MODEL_LABEL[model]} (t+1)",
                             mode="lines", line=dict(color=c[MODEL_COLOR[model]], width=1.6),
                             hovertemplate="%{y:.2f} €/MWh"))
    _event_bands(fig, theme, tp["Trading_Day"].min(), tp["Trading_Day"].max(), labels=False)
    base_layout(fig, theme, height, margin=dict(l=8, r=16, t=36, b=8))
    fig.update_yaxes(title_text="€/MWh")
    return fig


def cost_bars(theme: str, costs: dict, height=240) -> go.Figure:
    c = T(theme)
    names = [m for m in ["Naive", "LightGBM", "Ridge Regression"] if m in costs]
    fig = go.Figure(go.Bar(
        x=[costs[m] for m in names], y=[MODEL_LABEL[m] for m in names], orientation="h",
        marker=dict(color=[c[MODEL_COLOR[m]] for m in names], cornerradius=3),
        text=[f"{costs[m]:,.0f} €".replace(",", ".") for m in names], textposition="outside",
        cliponaxis=False, textfont=dict(size=12, color=c["text2"]),
        hovertemplate="%{y}: %{x:,.0f} €<extra></extra>"))
    base_layout(fig, theme, height, legend=False, hovermode="closest", margin=dict(l=8, r=80, t=8, b=8))
    fig.update_layout(bargap=0.4)
    fig.update_xaxes(showgrid=True, rangemode="tozero", title_text="Coste estimado 30 días (€)")
    fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(size=12, color=c["text2"]))
    return fig


# ------------------------------------------------------------------
# SHAP
# ------------------------------------------------------------------
def shap_bar(theme: str, model: str, top_n=10, height=380) -> go.Figure:
    c = T(theme)
    imp = D.shap_importance(model).head(top_n).iloc[::-1]
    col = c[MODEL_COLOR[model]]
    fig = go.Figure(go.Bar(
        x=imp["MeanAbsSHAP"], y=imp["Label"], orientation="h", marker=dict(color=col, cornerradius=3),
        text=[f"{s*100:.0f} %" for s in imp["Share"]], textposition="outside", cliponaxis=False,
        textfont=dict(size=11, color=c["text2"]),
        hovertemplate="<b>%{y}</b><br>|SHAP| medio %{x:.2f} €/MWh<extra></extra>"))
    base_layout(fig, theme, height, legend=False, hovermode="closest", margin=dict(l=8, r=48, t=8, b=8))
    fig.update_layout(bargap=0.3)
    fig.update_xaxes(showgrid=True, title_text="|SHAP| medio (€/MWh)")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=c["text2"]))
    return fig


def shap_beeswarm(theme: str, model: str, top_n=8, height=420) -> go.Figure:
    c = T(theme)
    sv, X, _ = D.shap_values(model)
    feats = D.shap_importance(model)["Feature"].head(top_n).tolist()[::-1]
    rng = np.random.default_rng(7)
    fig = go.Figure()
    scale = [[0, c["div_neg"]], [0.5, c["div_mid"]], [1, c["div_pos"]]]
    for i, f in enumerate(feats):
        v = X[f].astype(float)
        norm = (v - v.quantile(0.05)) / max(v.quantile(0.95) - v.quantile(0.05), 1e-9)
        norm = norm.clip(0, 1)
        y = i + rng.uniform(-0.28, 0.28, len(v))
        fig.add_trace(go.Scattergl(
            x=sv[f], y=y, mode="markers", showlegend=False,
            marker=dict(size=5, color=norm, colorscale=scale, cmin=0, cmax=1, opacity=0.85,
                        showscale=(i == 0),
                        colorbar=dict(title=dict(text="Valor", side="right", font=dict(size=10, color=c["muted"])),
                                      tickvals=[0, 1], ticktext=["bajo", "alto"], thickness=8, len=0.6,
                                      outlinewidth=0, tickfont=dict(size=10, color=c["muted"]))),
            customdata=v, hovertemplate=f"<b>{D.feature_label(f)}</b><br>SHAP %{{x:.2f}} €/MWh<br>valor %{{customdata:.2f}}<extra></extra>"))
    fig.add_vline(x=0, line_width=1, line_color=c["axis"])
    base_layout(fig, theme, height, legend=False, hovermode="closest")
    fig.update_yaxes(tickvals=list(range(len(feats))), ticktext=[D.feature_label(f) for f in feats],
                     showgrid=False, tickfont=dict(size=11, color=c["text2"]))
    fig.update_xaxes(showgrid=True, title_text="Impacto en la predicción (€/MWh)")
    return fig


def shap_waterfall(theme: str, model: str, top_n=7, height=400) -> go.Figure:
    c = T(theme)
    contrib, base, pred = D.explain_next_day(model)
    head = contrib.head(top_n)
    rest = contrib.iloc[top_n:]["SHAP"].sum()
    labels = ["Valor base"] + head["Label"].tolist() + ["Resto de variables", "Predicción t+1"]
    values = [base] + head["SHAP"].tolist() + [rest, pred]
    measures = ["absolute"] + ["relative"] * (len(head) + 1) + ["total"]
    text = [f"{base:.1f}"] + [f"{v:+.2f}" for v in head["SHAP"]] + [f"{rest:+.2f}", f"{pred:.2f}"]
    text = [t.replace(".", ",") for t in text]
    fig = go.Figure(go.Waterfall(
        orientation="v", x=labels, y=values, measure=measures, text=text, textposition="outside",
        textfont=dict(size=11, color=c["text2"]), cliponaxis=False,
        connector=dict(line=dict(color=c["axis"], width=1)),
        increasing=dict(marker=dict(color=c["div_pos"])),
        decreasing=dict(marker=dict(color=c["div_neg"])),
        totals=dict(marker=dict(color=c["neutral"])),
        hovertemplate="%{x}<br>%{y:.2f} €/MWh<extra></extra>"))
    base_layout(fig, theme, height, legend=False, hovermode="closest", margin=dict(l=8, r=8, t=24, b=8))
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=11, color=c["text2"]))
    fig.update_yaxes(title_text="€/MWh")
    lo = min(np.cumsum(values[:-1]).min(), pred) * 0.9
    hi = max(np.cumsum(values[:-1]).max(), base) * 1.06
    fig.update_yaxes(range=[lo, hi])
    return fig
