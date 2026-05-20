# train.py
import os, pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def train_and_save():
    os.makedirs("models", exist_ok=True)

    # Download a small synthetic fraud dataset (public, no auth needed)
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/creditcard_small.csv"
    # We'll generate synthetic data instead — no external dependency
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        "amt":             np.random.exponential(150, n),
        "zip":             np.random.randint(10000, 99999, n),
        "lat":             np.random.uniform(25, 48, n),
        "long":            np.random.uniform(-125, -65, n),
        "city_pop":        np.random.randint(1000, 1_000_000, n),
        "merch_lat":       np.random.uniform(25, 48, n),
        "merch_long":      np.random.uniform(-125, -65, n),
        "trans_hour":      np.random.randint(0, 24, n),
        "trans_dayofweek": np.random.randint(0, 7, n),
        "trans_month":     np.random.randint(1, 13, n),
        "trans_day":       np.random.randint(1, 29, n),
        "age":             np.random.randint(18, 90, n),
        "amt_log":         np.log1p(np.random.exponential(150, n)),
        "distance_km":     np.random.exponential(50, n),
        "merchant":        np.random.choice(["MerchA","MerchB","MerchC"], n),
        "category":        np.random.choice([
            "grocery_pos","gas_transport","home","shopping_pos","kids_pets",
            "food_dining","personal_care","health_fitness","entertainment",
            "shopping_net","misc_net","grocery_net","travel","misc_pos"], n),
        "gender":          np.random.choice(["M","F"], n),
        "city":            np.random.choice(["Springfield","Portland","Austin"], n),
        "state":           np.random.choice(["NC","CA","TX","NY"], n),
        "job":             np.random.choice(["Engineer","Teacher","Doctor","Other"], n),
    })

    # Fraud label — higher prob for high amount + night + risky category
    fraud_prob = (
        (df["amt"] > 500).astype(float) * 0.3 +
        (df["trans_hour"] < 6).astype(float) * 0.2 +
        df["category"].isin(["shopping_net","misc_net","misc_pos"]).astype(float) * 0.2 +
        (df["distance_km"] > 200).astype(float) * 0.15
    ).clip(0, 0.95)
    df["is_fraud"] = (np.random.random(n) < fraud_prob).astype(int)

    # Encode categoricals
    encoders = {}
    for col in ["merchant", "category", "gender", "city", "state", "job"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="logloss")
    model.fit(X, y)

    threshold = 0.5

    with open("models/fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/threshold.pkl", "wb") as f:
        pickle.dump(threshold, f)
    with open("models/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    print("✅ Model trained and saved to models/")

if __name__ == "__main__":
    train_and_save()