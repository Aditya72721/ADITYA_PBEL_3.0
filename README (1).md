# 🛡️ Sentinel — AI-Powered Financial Fraud Detection System

Sentinel is a machine learning-based web application that analyzes financial transactions in real time and scores them for fraud risk. Built entirely in Python using Flask and scikit-learn, it demonstrates a complete end-to-end fraud detection pipeline — from data generation and model training to an interactive web interface.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![scikit--learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

Every transaction submitted through the web interface is passed to a trained **Random Forest Classifier**, which returns a fraud probability score (0–100%) along with a verdict:

| Risk Score | Verdict     | Meaning                                   |
|------------|-------------|--------------------------------------------|
| 0–50%      | LEGIT       | Low risk — transaction looks clean         |
| 50–80%     | SUSPICIOUS  | Medium risk — extra verification suggested |
| 80–100%    | FRAUD       | High risk — flag for manual review         |

---

## ✨ Features

- **Real-time fraud scoring** via a trained ML model
- **Interactive dashboard** with an animated risk gauge
- **"Try random sample"** button to instantly test the model with realistic dummy transactions
- **Smart feature engineering** — an `amount_ratio` feature (transaction amount vs. the user's own historical average) that prevents small, ordinary transactions from being falsely flagged
- Clean, dark-themed, fully responsive UI (no external frontend framework — plain HTML/CSS/JS)

---

## 🧠 How the Model Works

Since real banking data is private and not publicly available, the model is trained on a **synthetic but realistic transaction dataset** (30,000 samples) generated with carefully tuned statistical distributions that mimic real spending behavior.

**Features used:**
- `amount` — transaction amount
- `hour_of_day` — time of transaction (0–23)
- `txn_count_24h` — number of transactions in the last 24 hours
- `avg_amount_30d` — user's average spend over the last 30 days
- `amount_ratio` — engineered feature: `amount / avg_amount_30d` (detects unusual spending spikes)
- `is_foreign` — whether the transaction is international
- `is_weekend` — whether it occurred on a weekend
- `new_merchant` — whether the merchant is new to the user

**Model:** `RandomForestClassifier` (300 trees, max depth 8, `class_weight="balanced"` to handle class imbalance since fraud is rare in real-world data).

**Evaluation:** Precision, Recall, F1-score, and ROC-AUC (~0.99 on the held-out test set).

---

## 🗂️ Project Structure

```
fraud_detection_website/
├── app.py              # Flask backend — serves the site and handles predictions
├── train_model.py      # Generates synthetic data and trains the RandomForest model
├── model.pkl           # Pre-trained model (ready to use out of the box)
├── templates/
│   └── index.html      # Frontend — form, animated risk gauge, live transaction ticker
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Aditya72721/ADITYA_PBEL_3.0.git

# Install dependencies
pip install flask scikit-learn pandas numpy joblib
```

### Run the app

```bash
python app.py
```

Then open your browser at:

```
http://localhost:5000
```

### Retrain the model

If you'd like to regenerate the dataset and retrain the model from scratch:

```bash
python train_model.py
```

This will overwrite `model.pkl` with a freshly trained model.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn, pandas, NumPy
- **Frontend:** HTML5, CSS3, vanilla JavaScript

---

## 📈 Possible Future Improvements

- Batch fraud detection via CSV upload
- Persistent transaction history / database integration
- User authentication for multi-user dashboards
- Model retraining on real-world, labeled datasets (e.g. the Kaggle Credit Card Fraud dataset)
- Deployment to a production WSGI server (e.g. Gunicorn) with a live hosting platform

---

## ⚠️ Disclaimer

This project is built for **educational and demonstration purposes**. The model is trained on synthetic data, not real financial transactions, and should not be used for actual fraud detection in production systems without significant additional validation, real-world data, and security review.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
