# app.py — Streamlit Weather Dashboard with strict LLM ticker
# -----------------------------------------------------------
# pip install streamlit pandas boto3 joblib python-dotenv openai matplotlib numpy

import os, io, json
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)  # load .env into os.environ

import boto3
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from openai import OpenAI

# ===== BASIC PAGE CONFIG =====
st.set_page_config(layout="wide", page_title="Weather Forecast Dashboard")

# ===== S3 CONFIG =====
S3_BUCKET = "weather-ingest"
S3_HISTORY_KEY = "combined/owm_history.csv"
S3_ICON_PREFIX = "app/icons/"
MODEL_S3_BUCKET = "weather-ingest"
MODEL_S3_KEY_MODEL = "app/models/latest/best_model.joblib"
MODEL_S3_KEY_LABEL = "app/models/latest/label_encoder.joblib"

# Where train_xgb.py writes API-scored forecasts
PREDICTIONS_API_PREFIX = "app/predictions_api/"

# ===== OPENAI / LLM CONFIG =====
ANCHOR_MODEL = os.getenv("ANCHOR_MODEL", "gpt-4o-mini")
REQUIRE_LLM = True  # hard-fail if LLM can't be used

# ===== ICONS =====
ICON_MAP = {
    "clear": "clear.png", "clouds": "clouds.png", "drizzle": "drizzle.png",
    "dust": "dust.png", "fog": "fog.png", "haze": "haze.png", "mist": "mist.png",
    "rain": "rain.png", "smoke": "smoke.png", "snow": "snow.png", "thunderstorm": "thunderstorm.png",
}
DEFAULT_ICON = "unknown.png"

# ===== AWS CLIENTS =====
@st.cache_resource
def s3_client():
    return boto3.client("s3")
s3 = s3_client()

# ===== LOAD MODEL (used for the bottom strip only) =====
@st.cache_resource
def load_pipeline_and_label_encoder_from_s3(bucket: str, model_key: str, label_key: str):
    m = s3.get_object(Bucket=bucket, Key=model_key)["Body"].read()
    l = s3.get_object(Bucket=bucket, Key=label_key)["Body"].read()
    pipe = joblib.load(io.BytesIO(m))
    le = joblib.load(io.BytesIO(l))
    return pipe, le

pipe, le = load_pipeline_and_label_encoder_from_s3(MODEL_S3_BUCKET, MODEL_S3_KEY_MODEL, MODEL_S3_KEY_LABEL)

# ===== FEATURE UTILS =====
def to_city_slug(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in str(name)).strip("-")

@st.cache_data(show_spinner=True)
def load_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    byts = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_csv(io.BytesIO(byts))

@st.cache_data
def fetch_icon_bytes(bucket: str, key: str):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:
        return None

def get_icon_bytes_for_label(label: str):
    fname = ICON_MAP.get(str(label).lower(), DEFAULT_ICON)
    key = f"{S3_ICON_PREFIX}{fname}"
    b = fetch_icon_bytes(S3_BUCKET, key)
    if b is None and fname != DEFAULT_ICON:
        b = fetch_icon_bytes(S3_BUCKET, f"{S3_ICON_PREFIX}{DEFAULT_ICON}")
    return b

def get_required_features_from_pipeline(pipeline) -> list[str]:
    cols = []
    pre = pipeline.named_steps.get("preprocess")
    if pre and hasattr(pre, "transformers_"):
        for _, _, sel in pre.transformers_:
            if isinstance(sel, list):
                cols.extend(sel)
    return cols

REQUIRED_FEATURES = get_required_features_from_pipeline(pipe)

def ensure_time_parts_if_needed(df: pd.DataFrame, dt_col: str = "DateTime") -> pd.DataFrame:
    if dt_col in df.columns:
        dt = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
        if "month" in REQUIRED_FEATURES and "month" not in df.columns: df["month"] = dt.dt.month
        if "day"   in REQUIRED_FEATURES and "day"   not in df.columns: df["day"] = dt.dt.day
        if "hour"  in REQUIRED_FEATURES and "hour"  not in df.columns: df["hour"] = dt.dt.hour
    return df

def ensure_datetime_from_parts(
    df: pd.DataFrame,
    year_col_candidates=("year", "Year", "YEAR"),
    month_col="month", day_col="day", hour_col="hour", out_col="DateTime",
) -> pd.DataFrame:
    if out_col in df.columns: return df
    cols_lower = {c.lower() for c in df.columns}
    if not {month_col.lower(), day_col.lower(), hour_col.lower()}.issubset(cols_lower): return df
    d = df.copy()
    def _real(name_lower: str) -> str: return next(c for c in d.columns if c.lower() == name_lower)
    m  = pd.to_numeric(d[_real(month_col.lower())], errors="coerce").fillna(1).astype(int).clip(1, 12)
    dd = pd.to_numeric(d[_real(day_col.lower())],   errors="coerce").fillna(1).astype(int).clip(1, 31)
    h  = pd.to_numeric(d[_real(hour_col.lower())],  errors="coerce").fillna(0).astype(int).clip(0, 23)
    y = None
    for cand in year_col_candidates:
        if cand in d.columns or cand.lower() in cols_lower:
            yname = cand if cand in d.columns else _real(cand.lower())
            y = pd.to_numeric(d[yname], errors="coerce")
            break
    if y is None: y = pd.Series(pd.Timestamp.utcnow().year, index=d.index)
    y = y.fillna(pd.Timestamp.utcnow().year).astype(int).clip(1970, 2100)
    d[out_col] = pd.to_datetime({"year": y, "month": m, "day": dd, "hour": h}, errors="coerce", utc=True)
    return d

# ===== PLOTS =====
def make_pretty_plot_df_3h(df: pd.DataFrame, date_col: str = "DateTime", days: int = 7) -> pd.DataFrame:
    """
    Return the last `days` aggregated to 3-hour cadence.
    Keeps good shape (8 pts/day) without noisy hour-by-hour jitter.
    """
    if date_col not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce", utc=True)
    d = d.dropna(subset=[date_col]).sort_values(date_col)

    end = d[date_col].max()
    start = end - pd.Timedelta(days=days)
    d = d[(d[date_col] >= start) & (d[date_col] <= end)]

    # Set index and resample to 3-hour means (handles duplicates/sub-hour data)
    d = d.set_index(date_col)

    cols = [c for c in ["Temperature", "Humidity", "Wind_Speed", "Pressure"] if c in d.columns]
    if not cols:
        return d.reset_index()

    d_res = d[cols].resample("3H").mean().dropna(how="all")
    return d_res.reset_index()



def plot_dual_axis_chart(data, temperature_col, humidity_col, date_col):
    fig, ax1 = plt.subplots()
    df = data.copy(); df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Temperature", color="tab:red")
    ax1.plot(df[date_col], df[temperature_col], color="tab:red", linewidth=2)
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.xaxis.set_major_locator(mdates.DayLocator()); ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    ax2 = ax1.twinx(); ax2.set_ylabel("Humidity", color="tab:blue")
    ax2.plot(df[date_col], df[humidity_col], color="tab:blue", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.2); plt.title("Temperative and Humidity Over Time")
    fig.tight_layout(); return fig

def plot_wind_speed_chart(data, date_col, wind_speed_col):
    fig, ax = plt.subplots()
    d = data.copy(); d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    ax.plot(d[date_col], d[wind_speed_col], color="tab:green", linewidth=2)
    ax.set_xlabel("Date"); ax.set_ylabel("Wind Speed"); ax.set_title("Wind Speed Over Time")
    ax.xaxis.set_major_locator(mdates.DayLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45); ax.grid(alpha=0.2); fig.tight_layout(); return fig

def plot_pressure_chart(data, date_col, pressure_col):
    fig, ax = plt.subplots()
    d = data.copy(); d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    ax.plot(d[date_col], d[pressure_col], color="tab:purple", linewidth=2)
    ax.set_xlabel("Date"); ax.set_ylabel("Pressure"); ax.set_title("Pressure Changes Over Time")
    ax.xaxis.set_major_locator(mdates.DayLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45); ax.grid(alpha=0.2); fig.tight_layout(); return fig

# ===== ON-THE-FLY PREDICTIONS FOR ICON STRIP =====
def predict_frame(frame: pd.DataFrame) -> pd.DataFrame:
    needed = get_required_features_from_pipeline(pipe)
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        st.error(f"Missing model features: {missing}"); st.stop()
    X = frame[needed].copy()
    proba = pipe.predict_proba(X)
    preds = le.inverse_transform(np.argmax(proba, axis=1))
    res = frame.copy(); res["predicted_weather"] = preds
    for i, cls in enumerate(le.classes_.tolist()):
        res[f"proba_{cls}"] = proba[:, i]
    return res

# ===== API PREDICTIONS (from train_xgb) =====
@st.cache_data(show_spinner=True)
def get_latest_api_predictions_manifest(bucket: str, base_prefix: str):
    prefix = base_prefix.rstrip("/") + "/run_id="
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in resp: return None
    mans = [o["Key"] for o in resp["Contents"] if o["Key"].endswith("/manifest.json")]
    if not mans: return None
    latest_key = max(mans)
    data = s3.get_object(Bucket=bucket, Key=latest_key)["Body"].read()
    man = json.loads(data.decode("utf-8"))
    start = latest_key.find("run_id=") + len("run_id="); end = latest_key.find("/", start)
    latest_run_id = latest_key[start:end]; man["run_id"] = man.get("run_id", latest_run_id)
    return man

@st.cache_data(show_spinner=True)
def load_city_api_predictions(bucket: str, base_prefix: str, city_name: str) -> pd.DataFrame | None:
    man = get_latest_api_predictions_manifest(bucket, base_prefix)
    if not man: return None
    run_id = man["run_id"]; slug = to_city_slug(city_name)
    key = f"{base_prefix.rstrip('/')}/city={slug}/run_id={run_id}/pred_from_api.csv"
    try:
        byts = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        dfp = pd.read_csv(io.BytesIO(byts))
        dfp = ensure_datetime_from_parts(dfp)
        if "DateTime" in dfp.columns:
            dfp["DateTime"] = pd.to_datetime(dfp["DateTime"], errors="coerce", utc=True)
            dfp = dfp.dropna(subset=["DateTime"]).sort_values("DateTime")
        return dfp
    except Exception:
        return None

# ===== OPENAI (STRICT) =====
def _get_openai_api_key():
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.getenv("OPENAI_API_KEY")

@st.cache_resource
def _openai_client():
    key = os.getenv("OPENAI_API_KEY")  # loaded by python-dotenv at app start
    if REQUIRE_LLM and not key:
        st.error(
            "OPENAI_API_KEY is missing. Put it in a .env file next to app.py like:\n"
            "OPENAI_API_KEY=sk-..."
        )
        st.stop()
    return OpenAI(api_key=key)

@st.cache_data(show_spinner=True, ttl=120)
def anchor_narration(model: str, payload: dict) -> str:
    """
    4–5 sentence anchor-style narration. Hard-fails if the LLM can't be used.
    """
    client = _openai_client()  # stops if key missing
    sys = (
        "You are a calm, professional TV weather anchor for a U.S. audience."
        " Write 4–5 concise sentences—no markdown, no emojis. "
        "Do not start with '<City> — <time>' or any timestamp. "
        "Begin with a present-tense summary of the current conditions in <City>, using numbers and units: temperature in °K, humidity %, wind in m/s, and pressure in hPa."
        "Then summarize the last week temperature range and the pressure trend, followed by the next 12-hour outlook using the most frequent predicted label and its share (e.g., 'Clouds likely for ~70% of hours')."
        "End with this exact pattern: Forecast at <forecast_time> shows <predicted_label>. Then add one short, proactive recommendation tailored to the conditions."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.4,
        max_tokens=220,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(payload)}
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        st.error("Weather narration unavailable (empty response).")
        st.stop()
    return text

def render_ticker(text: str):
    # Auto speed: ~0.12s per char (min 18s, max 60s)
    duration = int(min(60, max(18, len(text) * 0.12)))
    st.markdown(
        f"""
        <style>
          .wx-ticker-wrap {{
            background:#0f172a; color:#e2e8f0; border-radius:8px; padding:8px 0;
            overflow:hidden; border:1px solid rgba(148,163,184,.2); margin:8px 0 12px 0;
          }}
          .wx-ticker-inner {{
            display:inline-block; white-space:nowrap; padding-left:100%;
            animation: wx-ticker {duration}s linear infinite;
            font-size: 0.98rem;
          }}
          @keyframes wx-ticker {{ 0% {{ transform: translateX(0); }} 100% {{ transform: translateX(-100%); }} }}
        </style>
        <div class="wx-ticker-wrap"><div class="wx-ticker-inner">{text}</div></div>
        """,
        unsafe_allow_html=True,
    )

def pick_latest_or_selected(preds_df: pd.DataFrame):
    if preds_df is None or preds_df.empty: return None, None
    if "DateTime" in preds_df.columns:
        d = preds_df.copy()
        d["DateTime"] = pd.to_datetime(d["DateTime"], errors="coerce", utc=True)
        d = d.dropna(subset=["DateTime"]).sort_values("DateTime")
    else:
        return None, None
    sel = st.session_state.get("ts_picker_api")
    if sel:
        try:
            ts = pd.to_datetime(sel, utc=True)
            row = d.loc[d["DateTime"] == ts].head(1)
            if not row.empty: return ts, row.iloc[0]
        except Exception:
            pass
    ts = d["DateTime"].max()
    row = d.loc[d["DateTime"] == ts].head(1)
    return ts, (None if row.empty else row.iloc[0])

def extract_basic_metrics(source_row: pd.Series | None, hist_row: pd.Series | None):
    def g(obj, keys):
        for k in keys:
            if obj is not None and k in obj:
                try: return float(obj[k])
                except Exception:
                    try: return float(pd.to_numeric(obj[k], errors="coerce"))
                    except Exception: pass
        return None
    temp = g(source_row, ["Temperature", "main.temp"])
    hum  = g(source_row, ["Humidity", "main.humidity"])
    wind = g(source_row, ["Wind_Speed", "wind.speed"])
    pres = g(source_row, ["Pressure", "main.pressure"])
    if temp is None or hum is None or wind is None or pres is None:
        temp = temp or g(hist_row, ["Temperature", "main.temp"])
        hum  = hum  or g(hist_row, ["Humidity", "main.humidity"])
        wind = wind or g(hist_row, ["Wind_Speed", "wind.speed"])
        pres = pres or g(hist_row, ["Pressure", "main.pressure"])
    return temp, hum, wind, pres

def history_stats_last_hours(city_df: pd.DataFrame, hours: int = 24) -> dict:
    out = {}
    if "DateTime" not in city_df.columns or city_df.empty: return out
    d = city_df.copy()
    d["DateTime"] = pd.to_datetime(d["DateTime"], errors="coerce", utc=True)
    d = d.dropna(subset=["DateTime"]).sort_values("DateTime")
    end = d["DateTime"].max(); start = end - pd.Timedelta(hours=hours)
    part = d[(d["DateTime"] >= start) & (d["DateTime"] <= end)]
    if part.empty: return out
    def pick(*names):
        for n in names:
            if n in part.columns: return part[n]
        return pd.Series(dtype=float)
    temps = pick("Temperature", "main.temp")
    press = pick("Pressure", "main.pressure")
    hums  = pick("Humidity", "main.humidity")
    winds = pick("Wind_Speed", "wind.speed")
    if not temps.empty:
        out["temp_min"] = round(float(temps.min()), 1)
        out["temp_max"] = round(float(temps.max()), 1)
        out["temp_avg"] = round(float(temps.mean()), 1)
    if not hums.empty:  out["hum_avg"] = round(float(hums.mean()), 0)
    if not winds.empty: out["wind_max"] = round(float(winds.max()), 1)
    if not press.empty:
        delta = float(press.iloc[-1]) - float(press.iloc[0])
        out["pressure_trend"] = "rising" if delta > 1 else "falling" if delta < -1 else "steady"
    return out

def forecast_label_stats_next_hours(preds_df: pd.DataFrame, from_ts, hours: int = 12) -> dict:
    out = {}
    if preds_df is None or preds_df.empty or "DateTime" not in preds_df.columns: return out
    d = preds_df.copy()
    d["DateTime"] = pd.to_datetime(d["DateTime"], errors="coerce", utc=True)
    start = pd.to_datetime(from_ts, utc=True) if from_ts is not None else d["DateTime"].min()
    end = start + pd.Timedelta(hours=hours)
    window = d[(d["DateTime"] >= start) & (d["DateTime"] <= end)]
    if window.empty: return out
    if "predicted_weather" in window.columns:
        counts = window["predicted_weather"].value_counts()
        out["counts"] = counts.to_dict()
        out["top_label"] = counts.idxmax()
        out["top_label_share"] = round(float(counts.max() / counts.sum()), 2)
    return out

# ===== UI FLOW =====

# Load history
try:
    df = load_csv_from_s3(S3_BUCKET, S3_HISTORY_KEY)
except Exception as e:
    st.error(f"Failed to read `s3://{S3_BUCKET}/{S3_HISTORY_KEY}`\n{e}")
    st.stop()

# Friendly aliases
alias = {"main.temp":"Temperature", "main.humidity":"Humidity", "wind.speed":"Wind_Speed", "main.pressure":"Pressure"}
for old, new in alias.items():
    if old in df.columns and new not in df.columns:
        df[new] = df[old]

# Ensure DateTime exists
df = ensure_datetime_from_parts(df)

# City choices
city_col = "City_Name" if "City_Name" in df.columns else ("city_name" if "city_name" in df.columns else None)
if city_col is None:
    df["_city"] = "(all)"; city_col = "_city"
cities = sorted(df[city_col].dropna().unique().tolist())
default_city = cities[0] if cities else "(all)"
selected_city = st.session_state.get("city_choice", default_city)

# TITLE\
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0rem; /* Adjust this value as needed */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Weather Forecast Dashboard")

# TICKER (below title)
ticker_slot = st.empty()

# Build ticker content with current selection (session_state or default)
city_df = df[df[city_col] == selected_city].copy()
if "DateTime" in city_df.columns: city_df.sort_values("DateTime", inplace=True)

preds_df_for_ticker = load_city_api_predictions(S3_BUCKET, PREDICTIONS_API_PREFIX, selected_city)
ts, prow = pick_latest_or_selected(preds_df_for_ticker)
hist_row = city_df.tail(1).iloc[0] if len(city_df) else None
label = (str(prow.get("predicted_weather")) if prow is not None and "predicted_weather" in prow else "unknown")
temp, hum, wind, pres = extract_basic_metrics(prow, hist_row)
ts_label = ts.strftime("%Y-%m-%d %H:%M") if isinstance(ts, pd.Timestamp) else "latest"
hist_stats = history_stats_last_hours(city_df, hours=24)
fc_stats   = forecast_label_stats_next_hours(preds_df_for_ticker, ts, hours=12)

payload = {
    "city": selected_city,
    "time": ts_label,
    "now": {
        "predicted_label": label,
        "temp_C": temp, "humidity_pct": hum, "wind_m_s": wind, "pressure_hPa": pres
    },
    "history_24h": hist_stats,
    "forecast_12h": fc_stats
}
ticker_text = anchor_narration(ANCHOR_MODEL, payload)
with ticker_slot.container():
    render_ticker(ticker_text)

# CITY SELECTOR (under ticker)
selected_city = st.selectbox(
    "Select a City",
    options=cities,
    index=(cities.index(selected_city) if selected_city in cities else 0),
    key="city_choice"
)

# Re-slice after selection
city_df = df[df[city_col] == selected_city].copy()
if "DateTime" in city_df.columns: city_df.sort_values("DateTime", inplace=True)

# Charts frame (last 7 days)
plot_df = make_pretty_plot_df_3h(city_df, date_col="DateTime", days=7)

# LAYOUT
col1, col2, col3, col4 = st.columns([0.28, 0.28, 0.28, 0.16])

with col1:
    if {"Temperature", "Humidity", "DateTime"}.issubset(plot_df.columns):
        st.pyplot(plot_dual_axis_chart(plot_df, "Temperature", "Humidity", "DateTime"))

with col2:
    if {"Wind_Speed", "DateTime"}.issubset(plot_df.columns):
        st.pyplot(plot_wind_speed_chart(plot_df, "DateTime", "Wind_Speed"))

with col3:
    if {"Pressure", "DateTime"}.issubset(plot_df.columns):
        st.pyplot(plot_pressure_chart(plot_df, "DateTime", "Pressure"))

with col4:
    st.subheader("Forecast")
    preds_df = load_city_api_predictions(S3_BUCKET, PREDICTIONS_API_PREFIX, selected_city)
    if preds_df is None or preds_df.empty or "DateTime" not in preds_df.columns:
        st.info("No API-based forecasts found yet for this city.")
    else:
        ts_series = preds_df["DateTime"].sort_values().unique()
        ts_labels = [pd.to_datetime(t).strftime("%Y-%m-%d %H:%M") for t in ts_series]
        choice = st.selectbox("Pick a timestamp", options=ts_labels, index=0, key="ts_picker_api")
        selected_ts = ts_series[ts_labels.index(choice)]
        row = preds_df[preds_df["DateTime"] == selected_ts].head(1).iloc[0]
        sel_label = str(row.get("predicted_weather", "unknown"))
        icon_bytes = get_icon_bytes_for_label(sel_label)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if icon_bytes: st.image(icon_bytes, width=90)
        st.markdown(f"**{choice} → Predicted:** {sel_label}")

        # Update ticker to reflect the timestamp selection (LLM only)
        t_temp, t_hum, t_wind, t_pres = extract_basic_metrics(row, city_df.tail(1).iloc[0] if len(city_df) else None)
        t_hist_stats = history_stats_last_hours(city_df, hours=24)
        t_fc_stats   = forecast_label_stats_next_hours(preds_df, selected_ts, hours=12)
        t_payload = {
            "city": selected_city, "time": choice,
            "now": {"predicted_label": sel_label, "temp_C": t_temp, "humidity_pct": t_hum,
                    "wind_m_s": t_wind, "pressure_hPa": t_pres},
            "history_24h": t_hist_stats, "forecast_12h": t_fc_stats
        }
        with ticker_slot.container():
            render_ticker(anchor_narration(ANCHOR_MODEL, t_payload))

# ICON STRIP (historical, on-the-fly)
hist_preview = city_df.tail(200) if len(city_df) > 200 else city_df
hist_preview = hist_preview.copy()
hist_preview = ensure_time_parts_if_needed(hist_preview, dt_col="DateTime")
pred_df_hist = predict_frame(hist_preview) if not hist_preview.empty else pd.DataFrame()
if not pred_df_hist.empty and "DateTime" in pred_df_hist.columns:
    st.subheader("Previous weather (Last 40 hours)")
    strip = pred_df_hist[["DateTime", "predicted_weather"]].copy()
    strip["DateTime"] = pd.to_datetime(strip["DateTime"], errors="coerce")
    strip = strip.dropna(subset=["DateTime"]).tail(40)
    if len(strip) == 0:
        st.info("No valid timestamps to render.")
    else:
        cols = st.columns(len(strip))
        for i, (_, row) in enumerate(strip.iterrows()):
            with cols[i]:
                ib = get_icon_bytes_for_label(str(row["predicted_weather"]))
                if ib: st.image(ib, width=60, use_column_width=True)
                st.markdown("<div style='border-right: 5px dotted #999; height: 60px; margin: 4px 0;'></div>", unsafe_allow_html=True)
                ts_label = pd.to_datetime(row["DateTime"]).strftime("%Y-%m-%d %H:%M")
                st.markdown(
                    f"<div style='writing-mode:vertical-rl; transform: rotate(200deg); font-size: 12px;'>{ts_label}</div>",
                    unsafe_allow_html=True
                )
