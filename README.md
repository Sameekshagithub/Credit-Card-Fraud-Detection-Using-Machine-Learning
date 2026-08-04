# 💳 Credit Card Fraud Detection

An end-to-end machine learning pipeline that detects fraudulent credit card transactions using seven classification algorithms, SMOTE-based imbalance handling, and imbalance-aware evaluation metrics (Precision, Recall, F1, ROC-AUC, PR-AUC, MCC).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-brightgreen)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-purple)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview

Credit card fraud is a classic **extreme class imbalance** problem — fraudulent transactions typically make up less than 1–2% of all transactions. A naive model that predicts "legitimate" for every transaction would score ~98%+ accuracy while catching **zero fraud**, which makes this problem a great showcase for proper imbalanced-classification technique.

This repository walks through the **complete workflow**:

- Exploratory Data Analysis (EDA) on transaction patterns, amounts, and class distribution
- Feature scaling with `RobustScaler` (robust to the outlier-heavy `Amount` feature)
- Class imbalance handling via **SMOTE** (Synthetic Minority Over-sampling Technique)
- Training and comparing **7 algorithms**: Logistic Regression, K-Nearest Neighbors, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, and XGBoost
- Evaluation using metrics that actually matter on imbalanced data (not just accuracy)
- Feature importance analysis for tree-based models
- 5-fold stratified cross-validation for robustness
- Saving the best-performing model for reuse/deployment

---

## 📁 Repository Structure

```
credit-card-fraud-detection/
├── Credit_Card_Fraud_Detection.ipynb   # Main notebook — fully executed with outputs
├── creditcard.csv                      # Dataset (15,000 transactions, 31 columns)
├── README.md                           # You are here
├── best_model_*.pkl                    # Saved best model (generated on run)
└── amount_time_scaler.pkl              # Saved scaler (generated on run)
```

---

## 📊 Dataset

`creditcard.csv` follows the same schema as the well-known Kaggle "Credit Card Fraud Detection" dataset:

| Column | Description |
|---|---|
| `Time` | Seconds elapsed between this transaction and the first in the dataset |
| `V1`–`V28` | Anonymized/PCA-transformed features (protects sensitive transaction details) |
| `Amount` | Transaction amount |
| `Class` | Target — `0` = legitimate, `1` = fraud |

**Stats:**
- 15,000 total transactions
- 200 fraudulent (**~1.33%** fraud rate) — a realistic, highly imbalanced class distribution
- No missing values

> ℹ️ **Note on the dataset:** the shipped `creditcard.csv` is **synthetically generated** to match the statistical shape (feature distributions, imbalance ratio, PCA-like structure) of the original Kaggle dataset, so this project can be freely shared, cloned, and run without any data-licensing restrictions. The full pipeline is written against the same column schema, so you can drop in the real Kaggle `creditcard.csv` (from [kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)) and everything runs unchanged.

---

## 🤖 Algorithms Compared

| Model | Notes |
|---|---|
| Logistic Regression | Fast, interpretable linear baseline |
| K-Nearest Neighbors | Distance-based, sensitive to feature scale |
| Gaussian Naive Bayes | Probabilistic baseline, assumes feature independence |
| Decision Tree | Simple, interpretable, prone to overfitting |
| Random Forest | Bagged ensemble of trees, strong general-purpose performer |
| Gradient Boosting | Sequential boosting ensemble |
| XGBoost | Optimized, regularized gradient boosting |

---

## 📈 Results (on this run's held-out test set)

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 1.000 | 0.980 | **0.990** | 1.000 | 1.000 |
| K-Nearest Neighbors | 1.000 | 0.960 | 0.980 | 0.980 | 0.961 |
| Random Forest | 0.980 | 0.960 | 0.970 | 1.000 | 0.998 |
| Naive Bayes | 0.926 | 1.000 | 0.962 | 1.000 | 1.000 |
| Gradient Boosting | 0.922 | 0.940 | 0.931 | 1.000 | 0.987 |
| XGBoost | 0.922 | 0.940 | 0.931 | 1.000 | 0.989 |
| Decision Tree | 0.672 | 0.900 | 0.769 | 0.946 | 0.549 |

> Full confusion matrices, ROC curves, precision-recall curves, and feature-importance plots are generated inline in the notebook. Results will vary slightly with the real Kaggle dataset and random seeds.

**Why not just use accuracy?** With ~1.3% fraud, accuracy is nearly meaningless — always predicting "legitimate" scores ~98.7%. **Recall** (catching actual fraud) and **PR-AUC** (precision-recall trade-off under imbalance) are the metrics that matter operationally.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib jupyter
```

### 3. Launch the notebook
```bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

Run all cells top to bottom — the notebook is self-contained and reproducible (`random_state=42` throughout).

---

## 🧠 Methodology Highlights

- **Stratified train/test split** (75/25) to preserve the fraud ratio in both sets.
- **SMOTE applied only to the training set** — never to test data — to avoid data leakage and inflated metrics.
- **`class_weight='balanced'`** used as a complementary/fallback strategy for models where it applies.
- **RobustScaler** for `Time`/`Amount` since these are the only non-PCA-transformed, outlier-prone features.
- **Stratified 5-fold cross-validation** on the best model to confirm results generalize beyond a single split.

---

## 🔭 Future Improvements

- [ ] Deep learning approaches: autoencoder-based anomaly detection, feed-forward neural nets
- [ ] Hyperparameter tuning with `GridSearchCV` / `Optuna`
- [ ] Cost-sensitive learning with a custom business cost matrix (false negatives ≫ false positives in cost)
- [ ] Model explainability via SHAP values
- [ ] REST API deployment (FastAPI/Flask) for real-time transaction scoring
- [ ] Streaming/online learning for production fraud pipelines

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a PR.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## ⭐ Acknowledgements

Schema inspired by the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (ULB Machine Learning Group). Dataset in this repo is synthetically generated to match its statistical structure.

---

If this project helped you, consider giving it a ⭐ on GitHub!

