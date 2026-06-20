import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="AI Dropout Predictor", layout="wide")
st.title("🎓 AI Based Dropout Prediction System")

API_URL = "https://your-api.onrender.com/predict"

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    student_id = st.selectbox("Select Student", df['StudentID'])
    if st.button("Predict"):
        row = df[df['StudentID'] == student_id].iloc[0].to_dict()
        features = {k: v for k, v in row.items() if k not in ['StudentID', 'Target']}
        response = requests.post(API_URL, json=features)
        if response.status_code == 200:
            result = response.json()
            st.metric("Risk Probability", f"{result['probability']*100:.1f}%")
            st.info(f"Risk Level: {result['risk_level']}")
        else:
            st.error("API Error!")
