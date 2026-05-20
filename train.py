import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/fraudTrain.csv")
print("✅ Data loaded:", df.shape)
print("Fraud rate:", df["is_fraud"].mean().round(4))

df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["trans_hour"]      = df["trans_date_trans_time"].dt.hour
df["trans_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
df["trans_month"]     = df["trans_date_trans_time"].dt.month
df["trans_day"]       = df["trans_date_trans_time"].dt.day
df["age"]             = pd.to_datetime("today").year - pd.to_datetime(df["dob"]).dt.year
df["amt_log"]         = np.log1p(df["amt"])
df["distance_km"]     = np.sqrt((df["lat"]-df["merch_lat"])**2 + (df["long"]-df["merch_long"])**2) * 111

encoders = {}
for col in ["merchant","category","gender","city","state","job"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

FEATURES = ["amt","zip","lat","long","city_pop","merch_lat","merch_long",
            "trans_hour","trans_dayofweek","trans_month","trans_day",
            "age","amt_log","distance_km","merchant","category","gender",
            "city","state","job"]

X = df[FEATURES]
y = df["is_fraud"]

# Calculate imbalance ratio automatically
neg = (y == 0).sum()
pos = (y == 1).sum()
ratio = int(neg / pos)
print(f"⚖️ Class ratio: {ratio}:1 — setting scale_pos_weight={ratio}")

print("⏳ Training model...")
model = XGBClassifier(
    n_estimators=200,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=ratio,
    max_depth=6,
    learning_rate=0.1,
)
model.fit(X, y)

os.makedirs("models", exist_ok=True)
with open("models/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/threshold.pkl", "wb") as f:
    pickle.dump(0.5, f)
with open("models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("✅ Model + encoders saved!")