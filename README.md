Credit Risk Prediction System
📌 Overview

Credit risk analysis is a cornerstone of the FinTech industry, used to estimate the probability of customer loan defaults.
This project applies Machine Learning techniques to predict default probability on a Kaggle credit card dataset, with a special focus on vintage analysis to avoid label leakage.

We benchmarked traditional models (Logistic Regression, Balanced Random Forest) against advanced boosting methods, and identified CatBoost as the most effective for risk-aware financial decision-making.

📊 Dataset

Source: Kaggle – Credit Risk Vintage Analysis

Files:

application_record.csv — customer demographics (age, income, family, employment)

credit_record.csv — monthly repayment status history

Target variable: default = 1 if any future record (last 6 months) shows severe delinquency (STATUS ∈ {2,3,4,5}), else 0.

⚙️ Tech Stack

Python (pandas, numpy, matplotlib, seaborn)

Scikit-Learn (Logistic Regression, preprocessing, metrics)

Imbalanced-Learn (Balanced Random Forest)

CatBoost (gradient boosting on categorical features, final model)

🧩 Project Workflow

Exploratory Data Analysis (EDA)

Distribution plots (income, age, gender)

Class imbalance visualization

Vintage Analysis Labeling

Features built from earlier months (MONTHS_BALANCE ≤ -6)

Default label built from future months (MONTHS_BALANCE > -6)

Feature Engineering

Demographics: AGE_YEARS, YEARS_EMPLOYED, INCOME_PER_MEMBER

Credit history: WORST_STATUS, BAD_RATIO, HIST_LEN, etc.

Binning: age groups, employment length

Modeling & Benchmarking

Logistic Regression (baseline with class weights)

Balanced Random Forest (ensemble for imbalance)

CatBoost with scale_pos_weight (final model)

Evaluation Metrics

ROC-AUC

Precision, Recall, F1-score

Precision–Recall AUC

🚀 Results
Model	ROC-AUC	PR-AUC	Precision	Recall	F1	Threshold
Logistic Regression	~0.59	~0.003	0.05	0.01	0.02	~0.95
Balanced Random Forest	~0.74	~0.09	0.22	0.22	0.22	~0.85
CatBoost (final)	0.992	0.303	0.35	0.38	0.37	0.977

📂 See results/
 for:

metrics_summary.csv

catboost_roc.png (ROC curve)

catboost_pr.png (Precision–Recall curve)

📂 Repository Structure
credit-risk-prediction/
│── data/            # (not uploaded, see Kaggle link)
│── notebooks/       # Jupyter notebook with full pipeline
│── results/         # Metrics, plots
│── LICENSE          # MIT License
│── README.md        # Project documentation

📖 References

Chang, V., Sivakulasingam, S., Wang, H., Wong, S.T., Ganatra, M.A., Luo, J. (2022).
Credit Risk Prediction Using Machine Learning and Deep Learning: A Study on Credit Card Customers.

Kaggle Dataset: Vintage Credit Risk Analysis

✍️ Maintained by Sai Nikitha N S R
