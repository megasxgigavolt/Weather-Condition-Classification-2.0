import os, io, json
import boto3
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(layout="wide", page_title="Weather Forecast Dashboard")


S3_BUCKET="weather-ingest"
S3_HISTORY_KEY="combined/owm_history.csv"
S3_ICON_PREFIX="app/icons/"
MODEL_S3_BUCKET="weather-ingest"
MODEL_S3_KEY_MODEL="app/models/latest/best_model.joblib"
MODEL_S3_KEY_LABEL="app/models/latest/label_encoder.joblib"


# -------- Env config --------
# S3_BUCKET         = os.getenv("S3_BUCKET", "weather-ingest")
# S3_HISTORY_KEY    = os.getenv("S3_HISTORY_KEY", "combined/owm_history.csv")
# S3_ICON_PREFIX    = os.getenv("S3_ICON_PREFIX", "app/icons/")

# # Model artifacts in S3 (default: latest/)
# MODEL_S3_BUCKET   = os.getenv("MODEL_S3_BUCKET", "weather-ingest")
# MODEL_S3_KEY_MODEL= os.getenv("MODEL_S3_KEY_MODEL", "app/models/latest/best_model.joblib")
# MODEL_S3_KEY_LABEL= os.getenv("MODEL_S3_KEY_LABEL", "app/models/latest/label_encoder.joblib")

ICON_MAP = {
    "clear": "clear.png",
    "clouds": "clouds.png",
    "drizzle": "drizzle.png",
    "dust": "dust.png",
    "fog": "fog.png",
    "haze": "haze.png",
    "mist": "mist.png",
    "rain": "rain.png",
    "smoke": "smoke.png",
    "snow": "snow.png",
    "thunderstorm": "thunderstorm.png",
}
DEFAULT_ICON = "unknown.png"

# -------- AWS --------
@st.cache_resource
def s3_client():
    return boto3.client("s3")
s3 = s3_client()

# -------- Load model + label encoder from S3 --------
@st.cache_resource
def load_pipeline_and_label_encoder_from_s3(bucket: str, model_key: str, label_key: str):
    m = s3.get_object(Bucket=bucket, Key=model_key)["Body"].read()
    l = s3.get_object(Bucket=bucket, Key=label_key)["Body"].read()
    pipe = joblib.load(io.BytesIO(m))
    le   = joblib.load(io.BytesIO(l))
    return pipe, le

pipe, le = load_pipeline_and_label_encoder_from_s3(MODEL_S3_BUCKET, MODEL_S3_KEY_MODEL, MODEL_S3_KEY_LABEL)

# -------- Helpers --------
@st.cache_data(show_spinner=True)
def load_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    byts = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_csv(io.BytesIO(byts))

@st.cache_data
def fetch_icon_bytes(bucket: str, key: str) -> bytes | None:
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:
        return None

def get_icon_bytes_for_label(label: str) -> bytes | None:
    fname = ICON_MAP.get(str(label).lower(), DEFAULT_ICON)
    key = f"{S3_ICON_PREFIX}{fname}"
    b = fetch_icon_bytes(S3_BUCKET, key)
    if b is None and fname != DEFAULT_ICON:
        b = fetch_icon_bytes(S3_BUCKET, f"{S3_ICON_PREFIX}{DEFAULT_ICON}")
    return b

def plot_dual_axis_chart(data, temperature_col, humidity_col, date_col):
    fig, ax1 = plt.subplots()
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Temperature")
    ax1.plot(df[date_col], df[temperature_col])
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Humidity")
    ax2.plot(df[date_col], df[humidity_col])

    plt.title("Temperature and Humidity Over Time")
    fig.tight_layout()
    return fig

def plot_line(df, date_col, y_col, title, y_label):
    fig, ax = plt.subplots()
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    ax.plot(d[date_col], d[y_col])
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.xticks(rotation=45)
    fig.tight_layout()
    return fig

def get_required_features_from_pipeline(pipeline) -> list[str]:
    cols = []
    pre = pipeline.named_steps.get("preprocess")
    if pre and hasattr(pre, "transformers_"):
        for _, _, sel in pre.transformers_:
            if isinstance(sel, list):
                cols.extend(sel)
    return cols

REQUIRED_FEATURES = get_required_features_from_pipeline(pipe)

# -------- UI --------
st.title("Weather Forecast Dashboard")
with st.sidebar:
    st.header("Data sources")
    st.write(f"History: `s3://{S3_BUCKET}/{S3_HISTORY_KEY}`")
    st.write(f"Model:   `s3://{MODEL_S3_BUCKET}/{MODEL_S3_KEY_MODEL}`")
    st.write(f"Icons:   `s3://{S3_BUCKET}/{S3_ICON_PREFIX}`")

# Load history (owm_history.csv)
try:
    df = load_csv_from_s3(S3_BUCKET, S3_HISTORY_KEY)
except Exception as e:
    st.error(f"Failed to read `s3://{S3_BUCKET}/{S3_HISTORY_KEY}`\n{e}")
    st.stop()

# Friendly column aliases if needed
alias = {
    "main.temp": "Temperature",
    "main.humidity": "Humidity",
    "wind.speed": "Wind_Speed",
    "main.pressure": "Pressure",
}
for old, new in alias.items():
    if old in df.columns and new not in df.columns:
        df[new] = df[old]

# If DateTime not present but month/day/hour exist, create one (best-effort)
if "DateTime" not in df.columns and {"month", "day", "hour"}.issubset(df.columns):
    df["DateTime"] = pd.to_datetime(
        dict(year=2024, month=df["month"], day=df["day"], hour=df["hour"]),
        errors="coerce"
    )

# City selection
city_col = "City_Name" if "City_Name" in df.columns else ("city_name" if "city_name" in df.columns else None)
if city_col is None:
    df["_city"] = "(all)"
    city_col = "_city"
cities = sorted(df[city_col].dropna().unique().tolist())
selected_city = st.selectbox("Select a City", options=cities)

city_df = df[df[city_col] == selected_city].copy()
if "DateTime" in city_df.columns:
    city_df.sort_values("DateTime", inplace=True)

# ---- Predictions ----
def predict_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_FEATURES if c not in frame.columns]
    if missing:
        st.error(f"Missing model features: {missing}")
        st.stop()

    X = frame[REQUIRED_FEATURES].copy()
    proba = pipe.predict_proba(X)
    preds = le.inverse_transform(np.argmax(proba, axis=1))

    res = frame.copy()
    res["predicted_weather"] = preds
    for i, cls in enumerate(le.classes_.tolist()):
        res[f"proba_{cls}"] = proba[:, i]
    return res

preview = city_df.tail(200) if len(city_df) > 200 else city_df
pred_df = predict_frame(preview) if not preview.empty else pd.DataFrame()

# ---- Charts ----
col1, col2, col3, col4 = st.columns([0.28, 0.28, 0.28, 0.16])

with col1:
    if {"Temperature", "Humidity", "DateTime"}.issubset(city_df.columns):
        st.pyplot(plot_dual_axis_chart(city_df, "Temperature", "Humidity", "DateTime"))
with col2:
    if {"Wind_Speed", "DateTime"}.issubset(city_df.columns):
        st.pyplot(plot_line(city_df, "DateTime", "Wind_Speed", "Wind Speed Over Time", "Wind Speed"))
with col3:
    if {"Pressure", "DateTime"}.issubset(city_df.columns):
        st.pyplot(plot_line(city_df, "DateTime", "Pressure", "Pressure Changes Over Time", "Pressure"))

with col4:
    st.subheader("Latest Prediction")
    if not pred_df.empty:
        latest = pred_df.iloc[-1]
        label = str(latest["predicted_weather"])
        icon_bytes = get_icon_bytes_for_label(label)
        if icon_bytes:
            st.image(icon_bytes, width=75)
        st.write(f"**Predicted:** {label}")
        # show top-3
        classes = le.classes_.tolist()
        probs = {c: float(latest.get(f"proba_{c}", 0.0)) for c in classes}
        top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for c, p in top3:
            st.write(f"- {c}: {p:.3f}")

# Icons strip (last ~5 days @ 3h cadence)
if not pred_df.empty and "DateTime" in pred_df.columns:
    st.subheader("Previous weather (icons)")
    strip = pred_df.tail(40)
    cols = st.columns(len(strip)) if len(strip) > 0 else []
    for i, (_, row) in enumerate(strip.iterrows()):
        with cols[i]:
            ib = get_icon_bytes_for_label(str(row["predicted_weather"]))
            if ib:
                st.image(ib, width=60, use_column_width=True)
            ts = pd.to_datetime(row["DateTime"])
            st.markdown(
                f"<div style='writing-mode:vertical-rl; transform: rotate(200deg); font-size: 12px;'>{ts.strftime('%Y-%m-%d %H:%M')}</div>",
                unsafe_allow_html=True
            )
