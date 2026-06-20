# Create folders
mkdir -p api frontend scripts models data

# Write requirements.txt
cat > requirements.txt << "EOF2"
flask==2.3.2
flask-cors==4.0.0
gunicorn==21.2.0
streamlit==1.25.0
pandas==2.0.3
numpy==1.25.2
scikit-learn==1.3.0
xgboost==2.0.0
joblib==1.3.2
EOF2

# Write api/app.py
cat > api/app.py << "EOF2"
import os
import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load("models/xgb_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    print("✅ Models loaded!")
except:
    model = None
    preprocessor = None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    X_transformed = preprocessor.transform(input_df)
    prob = model.predict_proba(X_transformed)[0][1]
    risk = "High" if prob >= 0.65 else "Moderate" if prob >= 0.35 else "Low"
    return jsonify({"probability": round(float(prob), 4), "risk_level": risk})

@app.route('/')
def home():
    return jsonify({"message": "AI Dropout Prediction API is running!"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
EOF2

# Write frontend/streamlit_app.py
cat > frontend/streamlit_app.py << "EOF2"
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
EOF2

# Write scripts/generate_data.py
cat > scripts/generate_data.py << "EOF2"
import pandas as pd
import numpy as np
import os
os.makedirs("data", exist_ok=True)
np.random.seed(42)
n = 1000
data = {
    "StudentID": [f"S{str(i).zfill(4)}" for i in range(1, n+1)],
    "GPA": np.round(np.random.uniform(1.5, 4.0, n), 2),
    "Attendance": np.random.randint(30, 100, n),
    "Fee_Status": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    "Semester": np.random.choice([1,2,3,4,5,6,7,8], n),
}
df = pd.DataFrame(data)
df["Target"] = 0
df.loc[(df["GPA"] < 2.5) & (df["Attendance"] < 60), "Target"] = 1
df.loc[(df["Fee_Status"] == 1) & (df["GPA"] < 3.0), "Target"] = 1
df.to_csv("data/student_data.csv", index=False)
print("✅ Data created!")
EOF2

# Write scripts/train_model.py
cat > scripts/train_model.py << "EOF2"
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

os.makedirs("models", exist_ok=True)
df = pd.read_csv("data/student_data.csv")
X = df.drop(columns=["StudentID", "Target"])
y = df["Target"]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), X.select_dtypes(include=['int64','float64']).columns),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), X.select_dtypes(include=['object']).columns)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_t = preprocessor.fit_transform(X_train)
X_test_t = preprocessor.transform(X_test)

model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100)
model.fit(X_train_t, y_train)

joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")
print("✅ Model trained and saved!")
EOF2

echo "📂 All files created successfully!"

# Generate Data
python3 scripts/generate_data.py

# Train Model
python3 scripts/train_model.py

# Push to GitHub
git add .
git commit -m "Complete AI Dropout Prediction Project"
git push

echo "🎉 ALL DONE! Check your GitHub!"
