# app.py
# Streamlit app: Prediction | Forecasting | Visualizations | Chatbot
# pip install: streamlit numpy pandas joblib tensorflow statsmodels scikit-learn matplotlib

import os, io, math, json, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Deep learning & forecasting
import tensorflow as tf
from tensorflow.keras.models import load_model
import statsmodels.api as sm

st.set_page_config(page_title="✈️ Flight Fare Intelligence", layout="wide")

# -------------------------
# Sidebar: Model paths
# -------------------------
st.sidebar.header("⚙️ Settings")
default_ann_path = "/workspaces/Vamsithesis/best_model.h5"
default_scaler_path = "/workspaces/Vamsithesis/scaler.pkl"
default_sarimax_path = "/workspaces/Vamsithesis/sarimax_fare_forecast.pkl"
ann_path = st.sidebar.text_input("ANN model (.h5) path", default_ann_path)
scaler_path = st.sidebar.text_input("Scaler (.pkl) path", default_scaler_path)
sarimax_path = st.sidebar.text_input("SARIMAX model (.pkl) path", default_sarimax_path)
st.sidebar.markdown("---")
st.sidebar.caption("Paths default to your Colab saves; change if needed.")

# -------------------------
# Caching loaders
# -------------------------
@st.cache_resource(show_spinner=False)
def load_ann_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ANN .h5 not found at {path}")
    return load_model(path, compile = False)

@st.cache_resource(show_spinner=False)
def load_scaler(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler .pkl not found at {path}")
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_sarimax(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"SARIMAX .pkl not found at {path}")
    return sm.load(path)

# -------------------------------------------
# Feature order (same as you trained the ANN)
# -------------------------------------------
FEATURE_ORDER = [
    "baseFare",
    "totalTravelDistance",
    "travelDuration",
    "elapsedDays",
    "seatsRemaining",
    "segmentsDistance",
    "segmentsDurationInSeconds",
    "isNonStop_bin",
    "isBasicEconomy_bin",
    "isRefundable_bin",
    "days_ahead",
    "flight_is_weekend",
]

# -------------------------
# Helpers
# -------------------------
def compute_days_ahead(search_date, flight_date):
    if search_date is None or flight_date is None:
        return None
    if isinstance(search_date, dt.date) and not isinstance(search_date, dt.datetime):
        search_date = dt.datetime.combine(search_date, dt.time())
    if isinstance(flight_date, dt.date) and not isinstance(flight_date, dt.datetime):
        flight_date = dt.datetime.combine(flight_date, dt.time())
    return max(0, (flight_date - search_date).days)

def weekend_flag(flight_date):
    if not flight_date:
        return 0
    if isinstance(flight_date, dt.datetime):
        d = flight_date.date()
    else:
        d = flight_date
    return 1 if d.weekday() >= 5 else 0

def to_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def batch_prepare_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a batch for ANN prediction. Either expects FEATURE_ORDER columns,
       or raw columns + dates to derive 'days_ahead' and 'flight_is_weekend'."""
    if all(c in df.columns for c in FEATURE_ORDER):
        X = df[FEATURE_ORDER].copy()
    else:
        base_needed = {
            "baseFare","totalTravelDistance","travelDuration","elapsedDays",
            "seatsRemaining","segmentsDistance","segmentsDurationInSeconds",
            "isNonStop_bin","isBasicEconomy_bin","isRefundable_bin"
        }
        missing = base_needed - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required base columns: {missing}")

        # Derive features if dates present
        if "searchDate" in df.columns and "flightDate" in df.columns:
            sd = pd.to_datetime(df["searchDate"], errors="coerce")
            fd = pd.to_datetime(df["flightDate"], errors="coerce")
            days = (fd - sd).dt.days.clip(lower=0).fillna(0).astype(int)
            wk = fd.dt.weekday.fillna(0).astype(int).apply(lambda d: 1 if d >= 5 else 0)
            df["days_ahead"] = days
            df["flight_is_weekend"] = wk
        else:
            df["days_ahead"] = df.get("elapsedDays", 0).fillna(0).astype(int)
            df["flight_is_weekend"] = 0

        X = df[FEATURE_ORDER].copy()

    for c in FEATURE_ORDER:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X.astype(np.float32)

# -------------------------
# UI Tabs
# -------------------------
st.title("✈️ Flight Fare Intelligence")
tab_pred, tab_fc, tab_viz, tab_chat = st.tabs(
    ["🔮 Prediction", "📈 Forecasting", "📊 Visualizations", "💬 Chatbot"]
)

# ======================
# TAB 1: Prediction
# ======================
with tab_pred:
    st.subheader("🔮 High-Fare Probability (ANN, .h5)")
    # init session state for quick-fill
    defaults = {
        "baseFare": 150.0,
        "totalTravelDistance": 800.0,
        "travelDuration": 120.0,
        "elapsedDays": 14,
        "seatsRemaining": 5,
        "segmentsDistance": 800.0,
        "segmentsDurationInSeconds": 7200,
        "isNonStop_bin": True,
        "isBasicEconomy_bin": False,
        "isRefundable_bin": False,
        "searchDate": dt.date.today(),
        "flightDate": dt.date.today() + dt.timedelta(days=14),
    }
    for k,v in defaults.items():
        st.session_state.setdefault(k, v)

    colL, colR = st.columns([1,1])
    with colL:
        st.markdown("**Inputs** (same features used for training)")
        baseFare = st.number_input("baseFare", min_value=0.0, value=st.session_state["baseFare"], step=10.0, key="baseFare")
        totalTravelDistance = st.number_input("totalTravelDistance (miles)", min_value=0.0, value=st.session_state["totalTravelDistance"], step=10.0, key="totalTravelDistance")
        travelDuration = st.number_input("travelDuration (minutes)", min_value=0.0, value=st.session_state["travelDuration"], step=5.0, key="travelDuration")
        elapsedDays = st.number_input("elapsedDays (from data)", min_value=0, value=st.session_state["elapsedDays"], step=1, key="elapsedDays")
        seatsRemaining = st.number_input("seatsRemaining", min_value=0, value=st.session_state["seatsRemaining"], step=1, key="seatsRemaining")
        segmentsDistance = st.number_input("segmentsDistance (miles)", min_value=0.0, value=st.session_state["segmentsDistance"], step=10.0, key="segmentsDistance")
        segmentsDurationInSeconds = st.number_input("segmentsDurationInSeconds", min_value=0, value=st.session_state["segmentsDurationInSeconds"], step=60, key="segmentsDurationInSeconds")

        isNonStop_bin = st.checkbox("Non-stop", value=st.session_state["isNonStop_bin"], key="isNonStop_bin")
        isBasicEconomy_bin = st.checkbox("Basic Economy", value=st.session_state["isBasicEconomy_bin"], key="isBasicEconomy_bin")
        isRefundable_bin = st.checkbox("Refundable", value=st.session_state["isRefundable_bin"], key="isRefundable_bin")

        st.markdown("**Dates (for derived features)**")
        sd = st.date_input("searchDate", value=st.session_state["searchDate"], key="searchDate")
        fd = st.date_input("flightDate", value=st.session_state["flightDate"], key="flightDate")

        # Demo quick-fill button
        if st.button("⚡ Apply Demo Inputs 1"):
            # alternate demo values
            st.session_state.update({
                "baseFare": 220.0,
                "totalTravelDistance": 1500.0,
                "travelDuration": 240.0,
                "elapsedDays": 21,
                "seatsRemaining": 3,
                "segmentsDistance": 1500.0,
                "segmentsDurationInSeconds": 14400,
                "isNonStop_bin": True,
                "isBasicEconomy_bin": False,
                "isRefundable_bin": True,
                "searchDate": dt.date.today(),
                "flightDate": dt.date.today() + dt.timedelta(days=21),
            })
            st.experimental_rerun()

    with colR:
        derived_days = compute_days_ahead(st.session_state["searchDate"], st.session_state["flightDate"])
        st.number_input("days_ahead", min_value=0, value=(derived_days or st.session_state["elapsedDays"]), step=1, key="days_ahead")
        st.selectbox("flight_is_weekend", options=[0,1],
                     index=weekend_flag(st.session_state["flightDate"]), key="flight_is_weekend")

        if st.button("🔎 Predict"):
            try:
                model = load_ann_model(ann_path)
                scaler = load_scaler(scaler_path)
                form_vals = {k: st.session_state[k] for k in [
                    "baseFare","totalTravelDistance","travelDuration","elapsedDays",
                    "seatsRemaining","segmentsDistance","segmentsDurationInSeconds",
                    "isNonStop_bin","isBasicEconomy_bin","isRefundable_bin",
                    "days_ahead","flight_is_weekend"
                ]}
                x = batch_prepare_from_df(pd.DataFrame([form_vals]))
                x_s = scaler.transform(x)
                prob = float(model.predict(x_s, verbose=0).ravel()[0])
                st.success(f"High-fare probability: **{prob:.3f}**")
            except Exception as e:
                st.error(str(e))

        st.markdown("**Batch CSV Prediction**")
        st.caption("CSV with columns either FEATURE_ORDER or raw + dates. Outputs probabilities.")
        up = st.file_uploader("Upload CSV for batch", type=["csv"])
        if up is not None:
            try:
                df_in = pd.read_csv(up)
                X = batch_prepare_from_df(df_in)
                model = load_ann_model(ann_path)
                scaler = load_scaler(scaler_path)
                Xs = scaler.transform(X)
                probs = model.predict(Xs, verbose=0).ravel()
                out = df_in.copy()
                out["high_fare_prob"] = probs
                st.dataframe(out.head())
                st.download_button("📥 Download predictions CSV",
                                   data=out.to_csv(index=False).encode("utf-8"),
                                   file_name="predictions.csv",
                                   mime="text/csv")
            except Exception as e:
                st.error(str(e))

# ======================
# TAB 2: Forecasting
# ======================
with tab_fc:
    st.subheader("📈 Daily Fare Forecast (SARIMAX, .pkl)")
    # inputs
    if "fc_horizon" not in st.session_state:
        st.session_state["fc_horizon"] = 30

    c1, c2 = st.columns([1,1])
    with c1:
        horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=90,
                                  value=st.session_state["fc_horizon"], step=1, key="fc_horizon")

        # Demo quick-fill for forecast
        if st.button("⚡ Apply Demo Forecast 1"):
            st.session_state["fc_horizon"] = 30
            st.experimental_rerun()

        st.caption("Model forecasts daily average fare (global).")

    with c2:
        if st.button("🚀 Run Forecast"):
            try:
                forecaster = load_sarimax(sarimax_path)  # statsmodels results object
                fc = forecaster.get_forecast(steps=int(st.session_state["fc_horizon"])).predicted_mean
                fc = fc.rename("forecast").to_frame()
                st.line_chart(fc, height=280, use_container_width=True)
                st.dataframe(fc.tail(10))
                st.download_button("📥 Download forecast CSV",
                                   data=fc.to_csv().encode("utf-8"),
                                   file_name="forecast.csv",
                                   mime="text/csv")
            except Exception as e:
                st.error(str(e))

# ======================
# TAB 3: Visualizations
# ======================
with tab_viz:
    st.subheader("📊 Quick Visualizations (upload a small CSV sample)")
    up2 = st.file_uploader("Upload CSV (sample, e.g., few MB) for ad-hoc charts", type=["csv"], key="viz_upl")
    if up2 is not None:
        try:
            dsv = pd.read_csv(up2)
            # light coercions
            for col in ["totalFare","baseFare","totalTravelDistance","travelDuration","elapsedDays","seatsRemaining"]:
                if col in dsv.columns:
                    dsv[col] = pd.to_numeric(dsv[col], errors="coerce")
            st.write("Preview:", dsv.head())

            v1, v2 = st.columns(2)
            with v1:
                if "totalFare" in dsv.columns:
                    st.markdown("**Distribution: totalFare**")
                    st.bar_chart(dsv["totalFare"].dropna().clip(0, 2000).value_counts(bins=30).sort_index())
                if {"baseFare","totalFare"} <= set(dsv.columns):
                    st.markdown("**Scatter: baseFare vs totalFare**")
                    st.pyplot(plt.figure(figsize=(4,3)))
                    plt.scatter(dsv["baseFare"], dsv["totalFare"], s=4, alpha=0.3)
                    plt.xlabel("baseFare"); plt.ylabel("totalFare"); plt.tight_layout()
                    st.pyplot(plt.gcf()); plt.clf()
            with v2:
                if {"totalTravelDistance","totalFare"} <= set(dsv.columns):
                    st.markdown("**Scatter: distance vs totalFare**")
                    plt.figure(figsize=(4,3))
                    plt.scatter(dsv["totalTravelDistance"], dsv["totalFare"], s=4, alpha=0.3)
                    plt.xlabel("totalTravelDistance"); plt.ylabel("totalFare"); plt.tight_layout()
                    st.pyplot(plt.gcf()); plt.clf()
                if {"elapsedDays","totalFare"} <= set(dsv.columns):
                    st.markdown("**Trend: avg fare by elapsedDays**")
                    tmp = dsv.groupby("elapsedDays")["totalFare"].mean().reset_index()
                    st.line_chart(tmp.set_index("elapsedDays"))

        except Exception as e:
            st.error(str(e))
    else:
        st.info("Upload a small CSV to explore quick charts.")

# ======================
# TAB 4: Chatbot
# ======================
with tab_chat:
    st.subheader("💬 Helper Chat")
    st.caption("A lightweight helper to explain app usage and feature meanings.")
    if "chat_log" not in st.session_state:
        st.session_state["chat_log"] = []

    q = st.text_input("Ask something about this app or the dataset:")
    if st.button("Send"):
        if q.strip():
            # super-simple local responses (no external API)
            a = "I can help with predictions (ANN .h5), forecasting (SARIMAX .pkl), and quick visualizations.\n"
            if "feature" in q.lower():
                a += "Features: " + ", ".join(FEATURE_ORDER)
            elif "predict" in q.lower():
                a += "Go to Prediction tab, fill inputs or click 'Apply Demo Inputs 1', then press Predict."
            elif "forecast" in q.lower():
                a += "Go to Forecasting tab, set horizon or click 'Apply Demo Forecast 1', then Run Forecast."
            else:
                a += "Try asking: 'What inputs do predictions need?' or 'How to run a 30-day forecast?'."
            st.session_state["chat_log"].append(("You", q))
            st.session_state["chat_log"].append(("Bot", a))

    for who, msg in st.session_state["chat_log"][-8:]:
        if who == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")
