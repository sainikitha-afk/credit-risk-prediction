# Credit Risk Prediction System

## 📌 Overview
Credit risk analysis is crucial in the FinTech industry to evaluate the probability of loan default by customers.  
This project applies both **Machine Learning** and **Deep Learning** models to predict default probability on a Kaggle credit card dataset.  
We compare multiple models (benchmarking) to identify the most effective approach for risk-aware financial decision-making.  

Inspired by:  
Chang, V. et al., *Credit Risk Prediction Using Machine Learning and Deep Learning: A Study on Credit Card Customers* (2022).

---

## 📊 Dataset
- Source: [Kaggle – Credit Risk Vintage Analysis](https://www.kaggle.com/code/rikdifos/eda-vintage-analysis/data)  
- Contains customer demographics, credit history, and repayment information.  
- Target variable: Whether the customer **defaulted** or not.  

---

## ⚙️ Tech Stack
- **Python** (pandas, numpy, matplotlib, seaborn)  
- **Scikit-Learn** (Logistic Regression, Random Forest, etc.)  
- **XGBoost / LightGBM** (boosting models)  
- **TensorFlow / Keras** (Artificial Neural Networks)  
- **Imbalanced-learn** (for class imbalance handling, e.g., SMOTE)  

---

## 🧩 Project Workflow
1. **Exploratory Data Analysis (EDA)**  
   - Distribution of features  
   - Class imbalance visualization  
   - Correlation and multicollinearity checks  

2. **Data Preprocessing**  
   - Missing value imputation  
   - Feature scaling & encoding  
   - Handling imbalanced classes  

3. **Modeling & Benchmarking**  
   - Logistic Regression (baseline)  
   - Random Forest  
   - XGBoost / LightGBM  
   - Artificial Neural Network (ANN, 2–3 dense layers)  

4. **Evaluation Metrics**  
   - Accuracy  
   - ROC-AUC  
   - Precision, Recall, F1-score  
   - Confusion Matrix  

5. **Results & Insights**  
   - Compare ML vs DL performance  
   - Identify trade-offs (e.g., precision vs recall for default class)  

---

## 🚀 Results (to be updated after implementation)
- **XGBoost**: Expected to give highest ROC-AUC (~0.85).  
- **ANN**: May improve recall for minority class (defaults).  
- Benchmarking results will be uploaded in `/results/metrics.json`.  

---

## 📂 Repository Structure
credit-risk-prediction/
│── data/ # raw data (not uploaded, Kaggle link provided)
│── notebooks/ # Jupyter notebooks (EDA, model training)
│── src/ # Python scripts (preprocessing, training)
│── results/ # Metrics, plots, saved models
│── LICENSE # MIT License
│── README.md # Project documentation


---

## 📖 References
- Chang, V., Sivakulasingam, S., Wang, H., Wong, S.T., Ganatra, M.A., Luo, J. (2022).  
  *Credit Risk Prediction Using Machine Learning and Deep Learning: A Study on Credit Card Customers.*  
- Kaggle Dataset: [Vintage Credit Risk Analysis](https://www.kaggle.com/code/rikdifos/eda-vintage-analysis/data)

---

✍️ *Maintained by [Sai Nikitha N S R](https://github.com/sainikitha-afk)*  
