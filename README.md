# ✈️ XGBoost Classification Model – Airline Customer Satisfaction

## 📍 Project Overview
This project builds an **XGBoost classification model** to predict airline passenger satisfaction.  
It extends previous work using decision trees and random forests on the same dataset, allowing direct performance comparison and feature analysis.

By doing this, I demonstrate practical skills in:
- Data preprocessing and analysis
- Model training, hyperparameter tuning, and evaluation
- Feature importance analysis to interpret model results

---

## 📂 Dataset
- **Invistico_Airline.csv** – includes passenger demographics, flight details, and service ratings.
- Target variable: `satisfaction` (Satisfied / Neutral or Dissatisfied).

---

## ⚙️ Technologies & Libraries
- Python
- pandas, numpy, matplotlib
- scikit-learn
- XGBoost

---

## 🧰 Project Steps

### 1️⃣ Import libraries
```python
import numpy as np
import pandas as pd
import matplotlib as plt
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

from xgboost import XGBClassifier, plot_importance
2️⃣ Load data
python
Zkopírovat
Upravit
airline_data = pd.read_csv('Invistico_Airline.csv', error_bad_lines=False)
3️⃣ Build & tune model
Train an XGBoost classifier.

Use GridSearchCV to tune hyperparameters.

Evaluate using accuracy, precision, recall, and ROC-AUC.

4️⃣ Compare models
Compare XGBoost with previous decision tree and random forest models.

Choose the best-performing model.

5️⃣ Analyze feature importance
Identify which features most strongly influence passenger satisfaction.

✅ Results
The project concludes with:

A tuned XGBoost model with improved predictive performance.

Insights into the top features affecting customer satisfaction.

A clear comparison of XGBoost vs. decision tree and random forest models.

📌 Purpose
This project strengthens my applied machine learning skills and helps illustrate my ability to:

Build, tune, and evaluate advanced tree-based models.

Interpret and communicate model results for business decision-making.

⚡ This project was built as part of an ongoing exploration of tree-based models and boosting techniques in machine learning.

yaml
Zkopírovat
Upravit

---

✅ **Tip:**  
- Add a section at the end if you include visuals: confusion matrix, ROC curve, or feature importance plots.
- Use the GitHub preview to make sure formatting looks clean.

If you'd like, I can also make:
- A **short summary for LinkedIn**
- A **version with badges, usage instructions, and setup** for even more professional look.  

Let me know!
