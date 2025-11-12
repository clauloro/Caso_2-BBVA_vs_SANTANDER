# ======================================================
# BBVA vs Santander — Dashboard Financiero + Predicciones (LSTM / GRU / RNN)
# ======================================================

from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import json  # <-- para minmax_params.json

# ------------------ Config general ------------------
st.set_page_config(page_title="BBVA vs Santander — Dashboard", page_icon="📈", layout="wide")

# === 1) CSS global: ancho completo para TODA la app ===
st.markdown("""
<style>
.main .block-container { max-width: 100% !important; padding-left: 0 !important; padding-right: 0 !important; }
section.main > div { padding-left: 0 !important; padding-right: 0 !important; }
[data-testid="stVerticalBlock"] { padding-left: 1rem !important; padding-right: 1rem !important; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], .stTable { width: 100% !important; }
[data-baseweb="tab"] { max-width: none !important; }

.header-wrap {
  width: 100%;
  background: linear-gradient(90deg, #0b1120 0%, #1a2234 100%);
  padding: 2.2rem 1rem 2rem 1rem;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 2px 20px rgba(0,0,0,0.25);
  text-align: center;
}
.header-title {
  color: #f1f5f9; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.4px; margin-bottom: .6rem;
}
.header-line {
  width: 200px; height: 3px; margin: .4rem auto 1.1rem auto;
  background: linear-gradient(90deg, #3b82f6, #60a5fa); border-radius: 2px;
}
.header-sub {
  color: #cbd5e1; font-size: 1.06rem; line-height: 1.55rem; margin: 0 auto; max-width: 1100px;
}
.filters-bar { display: flex; gap: .8rem; align-items: end; flex-wrap: wrap; padding: 1rem 1rem 0.5rem 1rem; }
.filters-bar .cell { min-width: 220px; }
</style>
""", unsafe_allow_html=True)

# ---------- rutas ----------
BBVA_PATH = Path("data/interim/precios_limpios/BBVA_core_clean.csv")
SAN_PATH  = Path("data/interim/precios_limpios/SAN_core_clean.csv")

# ---------- colores ----------
BBVA_COLOR = "#1f77b4"
SAN_COLOR  = "#d62728"

# ---------- columnas importantes ----------
IMPORTANT = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# ======================================================
#                   UTILIDADES BÁSICAS
# ======================================================
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    lower_map = {c.lower(): c for c in df.columns}
    for col in IMPORTANT:
        candidates = [col, col.lower(), col.replace(" ", "_").lower(), col.replace(" ", "")]
        if col == "Adj Close":
            candidates += ["adjclose", "adj_close"]
        found = None
        for cand in candidates:
            if cand in df.columns:
                found = cand; break
            if cand in lower_map:
                found = lower_map[cand]; break
        if found:
            mapping[found] = col
    if mapping:
        df = df.rename(columns=mapping)
    return df

def _ensure_close_column(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_ohlcv(df)
    if "Close" in df.columns:
        return df
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric:
        df = df.rename(columns={numeric[-1]: "Close"})
    return df

def load_clean_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    else:
        cand = [c for c in df.columns if c.lower() in ("fecha", "date", "time", "datetime")]
        if cand:
            base = cand[0]
            df[base] = pd.to_datetime(df[base])
            df = df.sort_values(base).drop_duplicates(base).set_index(base)
        else:
            df.index = pd.to_datetime(df.index); df = df.sort_index()
    df = _ensure_close_column(df)
    return df

def price_chart_single(df: pd.DataFrame, title: str, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=title, line=dict(color=color)))
    fig.update_layout(
        title=title, margin=dict(l=10, r=10, t=40, b=10), height=480,
        legend=dict(orientation="h", y=1.05), xaxis=dict(title="Fecha", showgrid=False),
        yaxis=dict(title="Precio (€)", showgrid=True, gridcolor="#2a3342"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(11,18,32,1)", font=dict(color="#dbe4ff"),
    )
    return fig

def price_chart_both(bbva: pd.Series, san: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bbva.index, y=bbva.values, mode="lines", name="BBVA", line=dict(color=BBVA_COLOR)))
    fig.add_trace(go.Scatter(x=san.index,  y=san.values,  mode="lines", name="Santander", line=dict(color=SAN_COLOR)))
    fig.update_layout(
        title="BBVA vs Santander — Precio de cierre", margin=dict(l=10, r=10, t=40, b=10), height=520,
        legend=dict(orientation="h", y=1.05), xaxis=dict(title="Fecha", showgrid=False),
        yaxis=dict(title="Precio (€)", showgrid=True, gridcolor="#2a3342"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(11,18,32,1)", font=dict(color="#dbe4ff"),
    )
    return fig

def select_important(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in IMPORTANT if c in df.columns]
    return df[cols] if cols else df

def combined_table_with_prefix(df_bbva: pd.DataFrame, df_san: pd.DataFrame) -> pd.DataFrame:
    bbva_imp = select_important(df_bbva).add_prefix("BBVA_")
    san_imp  = select_important(df_san).add_prefix("SAN_")
    combined = bbva_imp.join(san_imp, how="inner")
    bbva_cols = [c for c in combined.columns if c.startswith("BBVA_")]
    san_cols  = [c for c in combined.columns if c.startswith("SAN_")]
    return combined[bbva_cols + san_cols]

def clip_by_dates(df: pd.DataFrame, d_from: date, d_to: date) -> pd.DataFrame:
    return df.loc[(df.index.date >= d_from) & (df.index.date <= d_to)]

# ======================================================
# 🔮 BLOQUE DE PREDICCIÓN (LSTM / GRU / RNN)
# ======================================================
READY_DIR     = Path("data/processed/ready_for_modeling")
MODELS_DIR    = Path("results/model_reports/models")
PREDS_DIR     = Path("results/model_reports/preds")
FORECASTS_DIR = Path("results/forecasts")
SUMMARY_PATH  = Path("results/model_reports/SUMMARY_top3_all_models.csv")

DEFAULT_BEST   = {"BBVA": "GRU", "SAN": "LSTM"}
LOOKBACK_DAYS  = 60  # zoom para el gráfico de predicción

# ====== Desescalado MinMax (si hay params) ======
MINMAX_PATH = READY_DIR / "minmax_params.json"

@st.cache_data
def load_minmax_params():
    if MINMAX_PATH.exists():
        with open(MINMAX_PATH, "r") as f:
            return json.load(f)
    return None

def inverse_minmax_series(series: pd.Series, col: str, mm: dict | None):
    if (mm is None) or (col not in mm):
        return series
    mn, mx = mm[col]["min"], mm[col]["max"]
    return series * (mx - mn) + mn

@st.cache_data
def load_summary():
    if SUMMARY_PATH.exists():
        df = pd.read_csv(SUMMARY_PATH)
        best = df.sort_values(["Ticker","Model","Test_RMSE"]).groupby(["Ticker","Model"], as_index=False).first()
        return best
    return pd.DataFrame(columns=["Ticker","Model","window"])

@st.cache_data
def get_best_window(ticker: str, model: str, fallback=10) -> int:
    best = load_summary()
    if not best.empty:
        row = best[(best["Ticker"]==ticker) & (best["Model"]==model)]
        if not row.empty:
            return int(row.iloc[0]["window"])
    return fallback

@st.cache_data
def load_ready_df(ticker: str) -> pd.DataFrame:
    path = READY_DIR / f"{ticker}_final_ready.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df

@st.cache_resource
def load_keras_model(path: Path):
    from tensorflow.keras.models import load_model
    return load_model(path, compile=False)

def make_sequences(df, window: int, target_col="Close"):
    vals = df.values.astype(np.float32)
    y_idx = df.columns.get_loc(target_col)
    X, y = [], []
    for s in range(len(df)-window):
        e = s + window
        X.append(vals[s:e, :])
        y.append(vals[e, y_idx])
    return np.stack(X), np.array(y, dtype=np.float32)

@st.cache_data
def temporal_split(df: pd.DataFrame, test_ratio=0.2):
    cut = int(len(df)*(1-test_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

@st.cache_data
def load_test_preds_csv(ticker: str, model: str):
    p = PREDS_DIR / f"{ticker}_{model}_test_preds.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["Date"])
    return pd.DataFrame()

def compute_test_preds(ticker: str, model: str, window: int) -> pd.DataFrame:
    df_ready = load_ready_df(ticker)
    if df_ready.empty or "Close" not in df_ready.columns:
        return pd.DataFrame()
    df_tr, df_te = temporal_split(df_ready, test_ratio=0.2)
    if len(df_te) <= window + 5:
        return pd.DataFrame()
    X_te, y_te = make_sequences(df_te, window=window, target_col="Close")
    path = MODELS_DIR / f"{ticker}_{model}_best.h5"
    if not path.exists():
        return pd.DataFrame()
    mdl = load_keras_model(path)
    y_pred = mdl.predict(X_te, verbose=0).ravel()
    dates = df_te.index[window:]
    return pd.DataFrame({"Date": dates, "y_true": y_te, "y_pred": y_pred})

@st.cache_data
def load_forecast_csv(ticker: str, model: str):
    p = FORECASTS_DIR / f"{ticker}_{model}_forecast_t5.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["Date"])
    return pd.DataFrame()

def plot_predictions_plotly(df_hist_close, df_pred, df_fore, title, color_hist):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist_close.index, y=df_hist_close.values, mode="lines",
                             name="Real (histórico/test)", line=dict(color=color_hist, width=2)))
    if not df_pred.empty:
        fig.add_trace(go.Scatter(x=df_pred["Date"], y=df_pred["y_pred"], mode="lines",
                                 name="Predicción (test)", line=dict(color="#FBBF24", width=2, dash="dash")))
    if not df_fore.empty:
        y_fc = df_fore["Forecast_real"] if "Forecast_real" in df_fore.columns else df_fore["Forecast_scaled"]
        fig.add_trace(go.Scatter(x=df_fore["Date"], y=y_fc, mode="lines+markers",
                                 name="Forecast t+1→t+5", line=dict(color="#22C55E", width=2)))
    fig.update_layout(title=title, height=520, margin=dict(l=10, r=10, t=45, b=10),
                      legend=dict(orientation="h", y=1.08),
                      xaxis=dict(title="Fecha", showgrid=False),
                      yaxis=dict(title="Precio (€)"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(11,18,32,1)", font=dict(color="#dbe4ff"))
    return fig

# ==== Helpers para leer el dataset exportado de zoom (si existe) ====
FORECASTS_EXPORT_DIRS = [Path("results/forecasts"), Path("../results/forecasts")]

def _find_zoom_export_path(ticker: str, model: str) -> Path | None:
    # Nombre estándar exportado por el notebook: {TICKER}_{MODEL}_zoom_plot_data.csv
    fname = f"{ticker}_{model}_zoom_plot_data.csv"
    for d in FORECASTS_EXPORT_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None

def load_zoom_export(ticker: str, model: str, use_real: bool = True) -> pd.DataFrame:
    """
    Devuelve un DataFrame largo con columnas:
    Date, Ticker, Model, series ∈ {real,pred_test,forecast}, value_scaled, value_real, value
    """
    p = _find_zoom_export_path(ticker, model)
    if p is None:
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["Date"])
    df["value"] = df["value_real"] if use_real else df["value_scaled"]
    return df

def plot_zoom_export_plotly(df_long: pd.DataFrame, ticker: str, model: str, lookback_days: int, use_real: bool = True) -> go.Figure:
    units = "€" if use_real else " (0–1)"
    title_suffix = "Precio real" if use_real else "Close escalado"
    fig = go.Figure()

    # Real (línea continua)
    dfr = df_long[df_long["series"] == "real"]
    fig.add_trace(go.Scatter(
        x=dfr["Date"], y=dfr["value"], mode="lines",
        name="Real (últimos)", line=dict(width=2)
    ))

    # Pred test (línea discontinua)
    dfp = df_long[df_long["series"] == "pred_test"]
    if not dfp.empty:
        fig.add_trace(go.Scatter(
            x=dfp["Date"], y=dfp["value"], mode="lines",
            name="Predicción test", line=dict(width=2, dash="dash")
        ))

    # Forecast (puntos + dash)
    dff = df_long[df_long["series"] == "forecast"]
    if not dff.empty:
        fig.add_trace(go.Scatter(
            x=dff["Date"], y=dff["value"], mode="lines+markers",
            name="Forecast t+1→t+5", line=dict(width=2, dash="dot"),
            marker=dict(size=7)
        ))

    fig.update_layout(
        title=f"{ticker} — {model} · Últimos {lookback_days} días + forecast 5 días ({title_suffix})",
        xaxis_title="Fecha",
        yaxis_title=f"Close{units}",
        legend=dict(orientation="h", y=1.06),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(11,18,32,1)", font=dict(color="#dbe4ff"),
    )
    return fig

def prediction_block_ui(ticker: str, color: str):
    st.markdown("### 🔮 Predicción (test + forecast)")
    default_model = DEFAULT_BEST.get(ticker, "LSTM")
    model = st.radio("Modelo", ["LSTM", "GRU", "RNN"],
                     index=["LSTM","GRU","RNN"].index(default_model),
                     key=f"{ticker}_model")
    window = get_best_window(ticker, model, fallback=10)

    colA, colB = st.columns([1,1])
    with colA:
        escala = st.radio("Escala", ["Real (€)", "Normalizada (0–1)"],
                          horizontal=True, key=f"{ticker}_escala")
    use_real = (escala == "Real (€)")

    # 1) Preferimos el dataset exportado de zoom (ya trae value_real si exportaste con mm)
    df_zoom = load_zoom_export(ticker, model, use_real=use_real)
    if not df_zoom.empty:
        fig = plot_zoom_export_plotly(df_zoom, ticker, model, LOOKBACK_DAYS, use_real=use_real)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Predicción test (tramo visible)")
            dfp = df_zoom[df_zoom["series"]=="pred_test"][["Date","value"]].rename(columns={"value":"Pred_test"})
            st.dataframe(dfp.tail(15) if not dfp.empty else pd.DataFrame({"Info":["No hay tramo visible"]}),
                         use_container_width=True, height=280)
        with c2:
            st.caption("Forecast t+1 → t+5")
            dff = df_zoom[df_zoom["series"]=="forecast"][["Date","value"]].rename(columns={"value":"Forecast"})
            st.dataframe(dff if not dff.empty else pd.DataFrame({"Info":["Sin forecast"]}),
                         use_container_width=True, height=280)
        return

    # 2) Fallback: cargar ready + preds + forecast y desescalar aquí si es posible
    df_ready = load_ready_df(ticker)
    if df_ready.empty or "Close" not in df_ready.columns:
        st.info("No se encontró el dataset escalado.")
        return

    df_pred = load_test_preds_csv(ticker, model)
    if df_pred.empty:
        df_pred = compute_test_preds(ticker, model, window)
    df_fore = load_forecast_csv(ticker, model)

    mm = load_minmax_params()

    # Serie histórica
    series_hist = df_ready["Close"].copy()

    # Si piden real, intentamos desescalar TODO
    if use_real:
        series_hist = inverse_minmax_series(series_hist, "Close", mm)
        if not df_pred.empty and "y_pred" in df_pred.columns:
            df_pred = df_pred.copy()
            df_pred["y_pred"] = inverse_minmax_series(df_pred["y_pred"], "Close", mm)
            if "y_true" in df_pred.columns:
                df_pred["y_true"] = inverse_minmax_series(df_pred["y_true"], "Close", mm)
        if not df_fore.empty:
            df_fore = df_fore.copy()
            if "Forecast_real" not in df_fore.columns and "Forecast_scaled" in df_fore.columns:
                df_fore["Forecast_real"] = inverse_minmax_series(df_fore["Forecast_scaled"], "Close", mm)
    else:
        # Normalizada: usamos Forecast_scaled si no hay real
        pass

    fig = plot_predictions_plotly(series_hist, df_pred, df_fore,
                                  f"{ticker} — {model} (window={window})", color)
    st.plotly_chart(fig, use_container_width=True)

    # Tablas con valores (en la escala seleccionada)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Predicción test (últimos 15)")
        if not df_pred.empty:
            cols = ["Date", "y_pred"] + (["y_true"] if "y_true" in df_pred.columns else [])
            st.dataframe(df_pred[cols].tail(15), use_container_width=True, height=280)
        else:
            st.info("Sin predicción de test disponible.")

    with c2:
        st.caption("Forecast t+1 → t+5")
        if not df_fore.empty:
            cols = ["Date"] + (["Forecast_real"] if use_real and "Forecast_real" in df_fore.columns else ["Date", "Forecast_scaled"][1:])
            st.dataframe(df_fore[cols], use_container_width=True, height=280)
        else:
            st.info("Sin forecast disponible.")

# ======================================================
#                   CARGA DE DATOS PRINCIPAL
# ======================================================
if not BBVA_PATH.exists() or not SAN_PATH.exists():
    st.error("No se encontraron los CSV en data/interim/precios_limpios/.")
    st.stop()

df_bbva = load_clean_csv(BBVA_PATH)
df_san  = load_clean_csv(SAN_PATH)

# ======================================================
#                   ENCABEZADO
# ======================================================
st.markdown("""
<div class="header-wrap">
  <div class="header-title">📊 BBVA vs Santander — Dashboard Financiero</div>
  <div class="header-line"></div>
  <div class="header-sub">
    Visualiza, compara y proyecta el rendimiento histórico y futuro de las dos entidades líderes del IBEX 35.
  </div>
</div>
""", unsafe_allow_html=True)

# Aviso si los exports de zoom están en un directorio vecino
if _find_zoom_export_path("BBVA", DEFAULT_BEST.get("BBVA","LSTM")) is None and Path("../results/forecasts").exists():
    st.info("Si exportaste los datasets de zoom desde el notebook en '../results/forecasts', la app los detectará automáticamente.")

# ======================================================
#                   BARRA DE FILTROS
# ======================================================
global_min = min(df_bbva.index.min(), df_san.index.min()).date()
global_max = max(df_bbva.index.max(), df_san.index.max()).date()
if "desde" not in st.session_state: st.session_state.desde = global_min
if "hasta" not in st.session_state: st.session_state.hasta = global_max

st.markdown('<div class="filters-bar">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([0.18, 0.18, 0.18, 0.12])

with c1:
    st.session_state.desde = st.date_input("Desde", st.session_state.desde, global_min, global_max, key="desde_input")
with c2:
    st.session_state.hasta = st.date_input("Hasta", st.session_state.hasta, global_min, global_max, key="hasta_input")
with c3:
    preset = st.selectbox("Atajos", ["—", "Todo", "Últimos 5 años", "Últimos 3 años", "Último año", "Desde 2023", "Desde 2024", "Desde 2025"], index=0)
with c4:
    if st.button("Aplicar atajo"):
        today = global_max
        if preset == "Todo": st.session_state.desde, st.session_state.hasta = global_min, global_max
        elif preset == "Últimos 5 años": st.session_state.desde, st.session_state.hasta = max(global_min, today - timedelta(days=1825)), today
        elif preset == "Últimos 3 años": st.session_state.desde, st.session_state.hasta = max(global_min, today - timedelta(days=1095)), today
        elif preset == "Último año": st.session_state.desde, st.session_state.hasta = max(global_min, today - timedelta(days=365)), today
        elif preset == "Desde 2023": st.session_state.desde, st.session_state.hasta = max(global_min, date(2023,1,1)), today
        elif preset == "Desde 2024": st.session_state.desde, st.session_state.hasta = max(global_min, date(2024,1,1)), today
        elif preset == "Desde 2025": st.session_state.desde, st.session_state.hasta = max(global_min, date(2025,1,1)), today
st.markdown('</div>', unsafe_allow_html=True)

desde = max(st.session_state.desde, global_min)
hasta = min(st.session_state.hasta, global_max)
if desde > hasta:
    st.warning("⚠️ Rango de fechas no válido.")
    st.stop()

df_bbva_f = clip_by_dates(df_bbva, desde, hasta)
df_san_f  = clip_by_dates(df_san, desde, hasta)

# ======================================================
#                   CONTENIDO PRINCIPAL
# ======================================================
tab1, tab2, tab3 = st.tabs(["BBVA", "Santander", "Ambos"])

with tab1:
    st.plotly_chart(price_chart_single(df_bbva_f, "BBVA — Precio de cierre", BBVA_COLOR), use_container_width=True)
    st.dataframe(select_important(df_bbva_f).tail(10), use_container_width=True, height=280)

    st.markdown("---")
    prediction_block_ui("BBVA", BBVA_COLOR)

with tab2:
    st.plotly_chart(price_chart_single(df_san_f, "Santander — Precio de cierre", SAN_COLOR), use_container_width=True)
    st.dataframe(select_important(df_san_f).tail(10), use_container_width=True, height=280)

    st.markdown("---")
    prediction_block_ui("SAN", SAN_COLOR)

with tab3:
    if df_bbva_f.empty or df_san_f.empty:
        st.warning("Faltan datos de alguno de los dos activos para la comparativa.")
    else:
        both_close = pd.concat(
            [df_bbva_f["Close"].rename("BBVA"), df_san_f["Close"].rename("Santander")],
            axis=1
        ).dropna()
        st.plotly_chart(
            price_chart_both(both_close["BBVA"], both_close["Santander"]),
            use_container_width=True
        )

        c1, c2 = st.columns(2)
        with c1:
            st.caption("BBVA — columnas clave")
            st.dataframe(select_important(df_bbva_f).tail(12), use_container_width=True, height=300)
        with c2:
            st.caption("Santander — columnas clave")
            st.dataframe(select_important(df_san_f).tail(12), use_container_width=True, height=300)

        st.markdown("---")
        st.caption("Tabla combinada (fechas comunes) · columnas clave con prefijos")
        combined = combined_table_with_prefix(df_bbva_f, df_san_f)
        st.dataframe(combined.tail(20), use_container_width=True, height=360)

# ======================================================
#                         FOOTER
# ======================================================
st.markdown(
    "<div style='text-align:center; margin: 30px 0; color:#94a3b8;'>"
    "BBVA vs Santander — Dashboard • Predicciones LSTM/GRU/RNN · © 2025</div>",
    unsafe_allow_html=True
)
