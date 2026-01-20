import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Traffic Control Dashboard",
    page_icon="🚦",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e293b, #020617);
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 6px 25px rgba(0,0,0,0.45);
    text-align: center;
}
.metric-title {
    color: #94a3b8;
    font-size: 14px;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🚦 AI Traffic Control Center")
st.caption("Real-time traffic signal optimization")
st.divider()

# ---------------- AUTO TRAFFIC DATA ----------------
def generate_traffic_state():
    now = datetime.now()

    return {
        "phase": np.random.randint(1, 5),
        "phase_duration_sec": np.random.uniform(20, 120),
        "avg_speed_kph": np.random.uniform(10, 70),
        "density_veh_per_km": np.random.uniform(20, 150),
        "pedestrian_count": np.random.randint(0, 80),
        "incident_flag": np.random.choice([0, 1]),
        "rain_mm": np.random.uniform(0, 20),
        "is_rush_hour": 1 if now.hour in [8, 9, 18, 19] else 0,
        "hour": now.hour,
        "queue_north": np.random.randint(5, 100),
        "queue_south": np.random.randint(5, 100),
        "queue_east": np.random.randint(5, 100),
        "queue_west": np.random.randint(5, 100),
    }

traffic = generate_traffic_state()

# ---------------- METRICS ----------------
total_queue = (
    traffic["queue_north"]
    + traffic["queue_south"]
    + traffic["queue_east"]
    + traffic["queue_west"]
)

c1, c2, c3, c4 = st.columns(4)

def metric(col, title, value):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

metric(c1, "🚗 Total Queue", total_queue)
metric(c2, "⚡ Avg Speed (km/h)", f"{traffic['avg_speed_kph']:.1f}")
metric(c3, "🚶 Pedestrians", traffic["pedestrian_count"])
metric(c4, "🌧 Rain (mm)", f"{traffic['rain_mm']:.1f}")

st.divider()

# ---------------- PREDICTION ----------------
X = pd.DataFrame([traffic])  # EXACT 13 FEATURES

reward = model.predict(X)[0]

left, right = st.columns([1, 2])

with left:
    st.subheader("🔮 AI Prediction")
    st.metric("Reward Score", f"{reward:.3f}")

    if reward > -1:
        st.success("🟢 Traffic Flow: OPTIMAL")
        st.progress(85)
    else:
        st.error("🔴 Traffic Flow: CONGESTED")
        st.progress(35)

    st.info("AI Recommendation: Dynamically adjust signal phase timing.")

with right:
    st.subheader("📊 Live Traffic State")
    st.dataframe(X, use_container_width=True)

# ---------------- HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.history.append({
    "Total Queue": total_queue,
    "Avg Speed": traffic["avg_speed_kph"],
})

history_df = pd.DataFrame(st.session_state.history)

st.subheader("📈 Traffic Trend")
st.line_chart(history_df)

# ---------------- AUTO REFRESH ----------------
st.caption("🔄 Auto-refreshing every 5 seconds")
time.sleep(5)
st.rerun()