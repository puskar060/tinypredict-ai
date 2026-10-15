import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Page Configuration
st.set_page_config(
    page_title="TinyPredict AI - Edge Industrial Monitoring & Repair Advisor",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ TinyPredict AI: Edge Predictive Maintenance & Repair System")
st.markdown("---")

# Sidebar for Configuration
st.sidebar.header("🎛️ Edge Device & Sensor Controls")
contamination_rate = st.sidebar.slider("Anomaly Sensitivity (Contamination)", 0.01, 0.20, 0.05)
simulated_vibration = st.sidebar.slider("Test Vibration Level (Hz)", 30.0, 100.0, 50.0)
simulated_temp = st.sidebar.slider("Test Temperature (°C)", 40.0, 110.0, 50.0)

# --- 1. Machine Learning Anomaly Detection Engine ---
@st.cache_resource
def train_anomaly_model(contamination):
    np.random.seed(42)
    # Baseline normal data: [Vibration, Temperature]
    normal_data = np.random.normal(loc=50.0, scale=2.0, size=(500, 2))
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(normal_data)
    return model

model = train_anomaly_model(contamination_rate)

# Predict status of live sensor input
live_input = np.array([[simulated_vibration, simulated_temp]])
prediction = model.predict(live_input)[0] # 1 is normal, -1 is anomaly
is_anomaly = True if prediction == -1 else False

# --- 2. Dashboard Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Real-time Edge Sensor Telemetry")
    st.metric(label="Vibration Reading", value=f"{simulated_vibration} Hz", delta=f"{simulated_vibration - 50:.1f} vs Baseline")
    st.metric(label="Temperature Reading", value=f"{simulated_temp} °C", delta=f"{simulated_temp - 50:.1f} vs Baseline")
    
    if not is_anomaly:
        st.success("🟢 Status: NORMAL - Machinery operating within safe parameters.")
    else:
        st.error("🔴 Status: ANOMALY DETECTED - Risk of imminent mechanical failure!")

with col2:
    st.subheader("🛠️ Automated RAG Repair & Maintenance Advisor")
    if not is_anomaly:
        st.info("No intervention required. System is running smoothly.")
    else:
        st.warning("⚠️ **Alert Triggered! Querying Technical Manual via RAG Agent...**")
        
        # Simulated RAG Knowledge Retrieval based on failure type
        if simulated_vibration > 75.0 and simulated_temp > 80.0:
            st.markdown("""
            **🔍 Diagnostic Result:** High Vibration & Thermal Overload in Main Bearing / Motor Shaft.
            
            **📋 Recommended Repair Steps (ISO-9001 Standard):**
            1. **Immediate Action:** Trigger emergency shutdown or reduce motor load by 50% within 2 minutes.
            2. **Inspection:** Check shaft alignment, bearing lubrication levels, and cooling fan status.
            3. **Replacement Parts:** Order *SKF Deep Groove Ball Bearing (Ref: 6205-2RS)* and high-temp synthetic grease.
            4. **Estimated Downtime:** 45 minutes. Estimated Cost: $180.
            """)
        else:
            st.markdown("""
            **🔍 Diagnostic Result:** Minor Anomaly / Unusual Friction Spike.
            
            **📋 Recommended Repair Steps:**
            1. Monitor sensor telemetry for the next 10 minutes.
            2. Check automated oil dispenser levels.
            """)

st.markdown("---")
st.caption("TinyPredict AI | Designed for Low-Cost Edge Devices (Raspberry Pi/Jetson Nano) | Optimized for Fully Funded Research & Industry 4.0")