"""Tokens de diseño compartidos por CSS y Plotly (modo oscuro / claro)."""

FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

THEMES = {
    "dark": {
        "page": "#0a0d13",
        "surface": "#11151d",
        "surface2": "#171c26",
        "text": "#e9ecf2",
        "text2": "#9aa3b2",
        "muted": "#6b7384",
        "grid": "#1d232e",
        "axis": "#2a3140",
        "neutral": "#8b93a3",       # serie de contexto (histórico, media UE)
        "neutral_soft": "rgba(139,147,163,0.14)",
        "neutral_bar": "#3a4252",
        "blue": "#3987e5",
        "orange": "#d95926",
        "aqua": "#199e70",
        "yellow": "#c98500",
        "violet": "#9085e9",
        "band": "rgba(255,255,255,0.035)",
        "band_label": "#7c8595",
        "div_neg": "#3987e5",
        "div_mid": "#2a2f3a",
        "div_pos": "#e66767",
        "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"],
        "good": "#0ca30c",
        "critical": "#e66767",
        "hover_bg": "#1b212c",
    },
    "light": {
        "page": "#f3f4f6",
        "surface": "#ffffff",
        "surface2": "#f7f8fa",
        "text": "#0f1623",
        "text2": "#4b5565",
        "muted": "#8a93a3",
        "grid": "#eceef2",
        "axis": "#d5d9e0",
        "neutral": "#6b7280",
        "neutral_soft": "rgba(107,114,128,0.10)",
        "neutral_bar": "#c5cad3",
        "blue": "#2a78d6",
        "orange": "#eb6834",
        "aqua": "#1baf7a",
        "yellow": "#eda100",
        "violet": "#4a3aa7",
        "band": "rgba(15,22,35,0.035)",
        "band_label": "#8a93a3",
        "div_neg": "#2a78d6",
        "div_mid": "#f0efec",
        "div_pos": "#e34948",
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"],
        "good": "#006300",
        "critical": "#d03b3b",
        "hover_bg": "#ffffff",
    },
}

# Identidad fija de cada entidad: el color sigue a la entidad, nunca al orden.
MODEL_COLOR = {"LightGBM": "blue", "Ridge Regression": "orange", "Naive": "aqua"}
MODEL_LABEL = {"LightGBM": "LightGBM", "Ridge Regression": "Ridge", "Naive": "Naïve"}


def T(theme: str) -> dict:
    return THEMES.get(theme or "dark", THEMES["dark"])
