import os

# Force single-threaded math libs BEFORE importing numpy/xgboost/sklearn.
# Prevents OpenMP/BLAS thread-pool deadlocks after gunicorn forks workers
# on CPU-limited instances (the cause of the /predict WORKER TIMEOUT).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load(os.path.join(BASE_DIR, "models", "xgb_model.pkl"))
    preprocessor = joblib.load(os.path.join(BASE_DIR, "models", "preprocessor.pkl"))
    # Pin XGBoost to a single thread to avoid fork/thread contention.
    try:
        model.set_params(n_jobs=1)
    except Exception:
        pass
    print("✅ Models loaded!")
except Exception as e:
    model = None
    preprocessor = None
    print(f"❌ Failed to load models: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        return jsonify({"error": "Model not loaded on server"}), 503
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400
        input_df = pd.DataFrame([data])
        X_transformed = preprocessor.transform(input_df)
        prob = model.predict_proba(X_transformed)[0][1]
        risk = "High" if prob >= 0.65 else "Moderate" if prob >= 0.35 else "Low"
        return jsonify({"probability": round(float(prob), 4), "risk_level": risk})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route('/')
def home():
    return jsonify({"message": "AI Dropout Prediction API is running!"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
