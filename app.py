import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Cancer Detection App", layout="wide")

# -----------------------------
# Custom CSS (Premium Look)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #ffffff;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stSlider label {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Data & Train Model
# -----------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

# -----------------------------
# HEADER
# -----------------------------
st.title("🧬 Cancer Classification System")
st.markdown("### AI-powered Tumor Detection (Benign vs Malignant)")

st.markdown("---")

# -----------------------------
# LAYOUT (2 columns)
# -----------------------------
col1, col2 = st.columns([1, 2])

# -----------------------------
# SIDEBAR INPUT (Left)
# -----------------------------
with col1:
    st.subheader("🔧 Input Features")

    input_data = []

    for feature in X.columns:
        val = st.slider(
            feature,
            float(X[feature].min()),
            float(X[feature].max()),
            float(X[feature].mean())
        )
        input_data.append(val)

    input_data = np.array(input_data).reshape(1, -1)

# -----------------------------
# RIGHT SIDE (OUTPUT)
# -----------------------------
with col2:
    st.subheader("📊 Patient Data")
    st.dataframe(pd.DataFrame(input_data, columns=X.columns))

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    st.markdown("---")

    st.subheader("🧠 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Benign Tumor (Non-Cancerous)")
    else:
        st.error("⚠️ Malignant Tumor (Cancerous)")

    st.subheader("📈 Prediction Probability")
    st.progress(float(prob[0][1]))

    st.write(f"Benign: {prob[0][1]:.2f}")
    st.write(f"Malignant: {prob[0][0]:.2f}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("🚀 Developed using Machine Learning & Streamlit")