import streamlit as st
import requests
import json

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance Demo",
    page_icon="🔧",
    layout="wide"
)

# ── API URL ───────────────────────────────────────────────────
API_URL = "https://ml-predictive-maintenance-1mum.onrender.com"

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🔧 Predictive Maintenance")
    st.markdown("---")
    st.markdown("### About This App")
    st.markdown(
        "This demo uses an XGBoost model trained on the "
        "**AI4I 2020 Predictive Maintenance Dataset** to predict "
        "CNC machine failures from sensor readings."
    )
    st.markdown("---")
    st.markdown("### Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("F1 Score", "0.81")
    col1.metric("Precision", "0.81")
    col2.metric("ROC-AUC", "0.98")
    col2.metric("Recall", "0.81")
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown("- **Source:** UCI ML Repository")
    st.markdown("- **Rows:** 10,000")
    st.markdown("- **Failure Rate:** 3.4%")
    st.markdown("- **Domain:** CNC Manufacturing")

# ── Main page ─────────────────────────────────────────────────
st.title("🏭 CNC Machine Failure Predictor")
st.markdown(
    "Adjust the sensor readings below and click **Predict** to check "
    "whether the machine is at risk of failure."
)
st.markdown("---")

# ── Input sliders ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Machine Settings")
    product_type = st.selectbox(
        "Product Quality Type",
        options=["L", "M", "H"],
        index=1,
        help="L = Low quality, M = Medium quality, H = High quality"
    )
    rotational_speed = st.slider(
        "Rotational Speed (RPM)",
        min_value=1000,
        max_value=3000,
        value=1551,
        step=10
    )
    torque = st.slider(
        "Torque (Nm)",
        min_value=3.0,
        max_value=80.0,
        value=42.8,
        step=0.1
    )
    tool_wear = st.slider(
        "Tool Wear (minutes)",
        min_value=0,
        max_value=260,
        value=0,
        step=1
    )

with col2:
    st.subheader("Temperature Readings")
    air_temperature = st.slider(
        "Air Temperature (K)",
        min_value=295.0,
        max_value=305.0,
        value=298.1,
        step=0.1
    )
    process_temperature = st.slider(
        "Process Temperature (K)",
        min_value=305.0,
        max_value=315.0,
        value=308.6,
        step=0.1
    )

    st.markdown("---")
    st.subheader("Engineered Features (auto-calculated)")
    temp_diff = process_temperature - air_temperature
    power = torque * rotational_speed
    st.metric("Temperature Difference (K)", f"{temp_diff:.2f}")
    st.metric("Mechanical Power (Nm·RPM)", f"{power:.0f}")

# ── Predict button ────────────────────────────────────────────
st.markdown("---")
predict_clicked = st.button("🔍 Predict", use_container_width=True, type="primary")

if predict_clicked:
    payload = {
        "type": product_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": float(rotational_speed),
        "torque": torque,
        "tool_wear": float(tool_wear)
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()

            st.markdown("---")
            st.subheader("Prediction Result")

            if result["prediction"] == 1:
                st.error(f"⚠️ **{result['result']}** — This machine is at risk!")
            else:
                st.success(f"✅ **{result['result']}** — This machine is operating normally.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", result["result"])
            col2.metric("Failure Probability", f"{result['failure_probability'] * 100:.2f}%")
            col3.metric("Confidence", f"{result['confidence'] * 100:.2f}%")

            st.markdown("#### Failure Probability")
            st.progress(result["failure_probability"])

        else:
            st.error(f"API Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the API. "
            "Make sure the FastAPI server is running with: `uvicorn api.main:app --reload`"
        )