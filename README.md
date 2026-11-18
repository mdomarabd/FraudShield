# 💳 FraudShield — Credit Card Fraud Detection

FraudShield is an end-to-end Machine Learning project designed to detect fraudulent credit card transactions in real-time, addressing severe class imbalance through feature engineering, threshold tuning, and ensemble modeling.

The project includes an interactive **Streamlit dashboard** for offline evaluation and live transaction scoring.

---

## 🚀 Key Features

- **Exploratory Data Analysis & Preprocessing**: Handling class imbalance and skewed distributions using robust scaling and feature extraction (`pca_magnitude`, `log_amount`).
- **Trained Models**:
  - **XGBoost (Tuned)**: Optimized for overall F1-score and precision while retaining high recall.
  - **Logistic Regression (Tuned)**: Prioritized for high recall (catching maximum fraud cases).
- **Interactive Streamlit Dashboard**:
  - Live metric computation on uploaded or sample datasets.
  - Transaction-level risk scoring and alert thresholds.
  - Confusion matrix visualization.

---

## 📊 Offline Model Evaluation (Hold-Out Test Set)

| Model | Accuracy | Recall (Fraud) | Precision (Fraud) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **99.85%** | 89.47% | **52.80%** | **0.664** | **0.981** |
| **Logistic Regression (Tuned)** | 97.11% | **90.53%** | 5.01% | 0.095 | 0.973 |

---

## 📂 Project Structure

```
ML_Project/
├── data/
│   ├── raw/
│   │   ├── README.md               # Instructions for downloading creditcard.csv
│   │   └── creditcard.csv          # Raw Kaggle dataset (git-ignored due to size)
│   └── processed/
│       └── streamlit_test_data.csv # Sample processed data for quick dashboard testing
├── models/
│   ├── logreg_fraud_model.pkl      # Trained Logistic Regression model
│   └── xgb_fraud_model.pkl         # Trained XGBoost model
├── app.py                          # Streamlit web application
├── model_metrics.csv               # Offline evaluation metrics
├── notebook.ipynb                  # Data exploration, preprocessing, & model training
├── random_streamlit_upload_with_class.csv # Demo dataset for dashboard upload
├── requirements.txt                # Python package dependencies
└── README.md
```

---

## 🛠️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/mdomarabd/FraudShield.git
cd FraudShield
```

### 2. Set up virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Download raw data
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it inside `data/raw/` if you want to rerun `notebook.ipynb`.

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📜 Dataset Reference
- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Transactions made by European cardholders in September 2013 (492 frauds out of 284,807 transactions).

