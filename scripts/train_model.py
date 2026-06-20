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
