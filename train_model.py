import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib, json

df = pd.read_csv("claims_data.csv")

# Encode categoricals
encoders = {}
for col in ["claim_type", "diagnosis", "category", "provider", "patient_gender"]:
    le = LabelEncoder()
    df[f"{col}_enc"] = le.fit_transform(df[col])
    encoders[col] = le
    joblib.dump(le, f"le_{col}.pkl")

# Save label classes
classes = {col: list(encoders[col].classes_) for col in encoders}
with open("encoder_classes.json","w") as f:
    json.dump(classes, f)

FEATURES = [
    "claim_type_enc","diagnosis_enc","category_enc","patient_age",
    "claim_amount_usd","days_since_submission","has_prior_auth",
    "is_chronic","missing_documents","is_readmission",
    "num_prev_claims","comorbidities"
]

X = df[FEATURES]
y = df["triage_score"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = GradientBoostingRegressor(n_estimators=300,max_depth=5,learning_rate=0.07,
                                   subsample=0.8,random_state=42)
model.fit(X_train,y_train)
preds = model.predict(X_test)
print(f"Triage model — MAE: {mean_absolute_error(y_test,preds):.2f}  R²: {r2_score(y_test,preds):.4f}")
joblib.dump(model,"triage_model.pkl")

# Anomaly / fraud model
iso = IsolationForest(n_estimators=200,contamination=0.08,random_state=42)
iso.fit(X_train)
joblib.dump(iso,"anomaly_model.pkl")
print("Models saved.")
