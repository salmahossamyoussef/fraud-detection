# 🛡️ Fraud Detection — Real-Time Transaction Analysis

**🌐 Live Demo:** [Click here to try the Streamlit App!](https://fraud-detection-8t2nltqd4seq8uwzvrdyag.streamlit.app/)


> A military-HUD styled fraud detection web app powered by XGBoost, built with Streamlit.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📸 Preview

The app features a dark, cyberpunk-inspired terminal UI with animated radar, CRT scanlines, and real-time threat level indicators. It accepts transaction parameters and outputs a fraud probability score along with a full risk signal breakdown.

---

## ✨ Features

- **Real-time fraud scoring** using a trained XGBoost classifier
- **Threat level classification**: NONE / LOW / MEDIUM / HIGH / CRITICAL
- **Animated circular gauge** showing fraud probability percentage
- **Risk signal analysis** across 5 factors: amount, hour, category, distance, age
- **Auto-training on first run** — no model file required at deployment
- **Full audit log** per prediction with timestamp and model metadata
- **Military HUD aesthetic** with CRT scanlines, radar sweep, and glow effects

---

## 🗂️ Project Structure

* 📓 **[Jupyter Notebook](./Notebook/):** Contains data preprocessing, exploratory data analysis (EDA), and model training.
* 🌐 **[Streamlit App](./app.py):** The production-ready code for the web application deployment.

```
fraud-detection/
├── app.py              # Main Streamlit application
├── train.py            # Model training script
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── fraud_model.pkl     # Trained fraud detection model
├── threshold.pkl       # Decision threshold parameters
└── encoders.pkl        # Data preprocessing encoders
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/salmahossamyoussef/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will **automatically train the model on first launch** and save it to the `models/` folder. Subsequent runs load the cached model instantly.

---

## ☁️ Deploy on Streamlit Cloud

1. Push your repo to GitHub (no need to include model files)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file path to `app.py`
4. Click **Deploy** — the model trains automatically on first startup

> The model is trained at runtime using synthetic fraud-pattern data, so no dataset file is needed in the repo.

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | XGBoost Classifier |
| Default Threshold | 0.5 (50%) |
| Training Data | Synthetic fraud-pattern dataset |
| Features | 19 (amount, category, hour, age, distance, geo, etc.) |
| Encoding | LabelEncoder per categorical column |

### Input Features

| Feature | Description |
|---------|-------------|
| `amt` | Transaction amount in USD |
| `category` | Merchant category (14 types) |
| `trans_hour` | Hour of transaction (0–23) |
| `age` | Cardholder age |
| `distance_km` | Distance to merchant |
| `gender` | Cardholder gender |
| + 13 more | Geo, time, and merchant metadata |

### Threat Levels

| Probability | Level |
|-------------|-------|
| ≥ 90% | 🔴 CRITICAL |
| ≥ 70% | 🟠 HIGH |
| ≥ 50% | 🟡 MEDIUM |
| ≥ 25% | 🟢 LOW |
| < 25% | ✅ NONE |

---

## 📦 Requirements

```
streamlit
xgboost
scikit-learn
pandas
numpy
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## ⚠️ Disclaimer

This app uses a **synthetically trained model** for demonstration purposes. It is not intended for use in real financial fraud prevention without retraining on actual labeled transaction data.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  <sub>FRAUD INTELLIGENCE ENGINE · XGBOOST CLASSIFIER · FOR AUTHORIZED USE ONLY</sub>
</div>