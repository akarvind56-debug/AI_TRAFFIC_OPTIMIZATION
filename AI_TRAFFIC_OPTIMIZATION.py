import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title= "AI TRAFFIC OPTIMIZATION" , layout= "centered")

st.title("AI TRAFFIC OPTIMIZATION APP")
st.write("Predict Traffic Reward")
 
st.header("Enter Traffic Parameters")

queue_north = st.number_input("Queue North", min_value=0)
queue_south = st.number_input("Queue South", min_value=0)
queue_east = st.number_input("Queue East", min_value=0)
queue_west = st.number_input("Queue West", min_value=0)

avg_speed_kph = st.number_input("avg_speed_kph" , min_value=0.0)
pedestrian_count = st.number_input("pedestrian_count", min_value=0)
phase_duration_sec = st.number_input("Phase Duration (sec)", min_value=0.0)

is_rush_hour = st.selectbox("Is Rush Hour?", [0, 1])
incident_flag = st.selectbox("Incident Present?", [0, 1])

if st.button("Predict Reward"):
    input_data = pd.DataFrame([{
        'queue_north': queue_north,
        'queue_south': queue_south,
        'queue_east': queue_east,
        'queue_west': queue_west,
        'avg_speed_kph': avg_speed_kph,
        'phase_duration_sec': phase_duration_sec,
        'is_rush_hour': is_rush_hour,
        'incident_flag': incident_flag,
        'pedestrian_count' : pedestrian_count
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"✅ Predicted Traffic Reward: {prediction:.3f}")

    if prediction > -1:
        st.info("🟢 Traffic flow is optimal")
    else:
        st.warning("🔴 Traffic congestion detected – optimize signal timing")