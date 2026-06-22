import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="AI Dropout Predictor", layout="wide")
st.title("🎓 AI Based Dropout Prediction System")

API_BASE = "https://ai-dropout-predictor-j3iv.onrender.com"
API_URL = f"{API_BASE}/predict"

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    student_id = st.selectbox("Select Student", df['StudentID'])
    if st.button("Predict"):
        row = df[df['StudentID'] == student_id].iloc[0].to_dict()
        features = {k: v for k, v in row.items() if k not in ['StudentID', 'Target']}
        try:
            with st.spinner("Contacting API (first request may take ~60s to wake the server)..."):
                # Warm up the free-tier server first
                try:
                    requests.get(f"{API_BASE}/health", timeout=120)
                except requests.exceptions.RequestException:
                    pass
                response = requests.post(API_URL, json=features, timeout=120)
            if response.status_code == 200:
                result = response.json()
                st.metric("Risk Probability", f"{result['probability']*100:.1f}%")
                st.info(f"Risk Level: {result['risk_level']}")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
        except requests.exceptions.Timeout:
            st.error("The server took too long to respond. It may be waking up. Please try again in a minute.")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to API: {e}")
