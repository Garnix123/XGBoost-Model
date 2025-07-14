✈️ XGBoost Classification Model – Airline Customer Satisfaction
📌 Overview
This project demonstrates how to build and evaluate an XGBoost classification model to predict airline passenger satisfaction. It builds on prior work with decision trees and random forests, allowing direct comparison of all three models to identify the best-performing approach.

By completing this project, I’ve deepened my practical skills in:

Data preprocessing and feature engineering

Training and tuning an XGBoost classifier

Evaluating model performance

Analyzing feature importance to understand key drivers of customer satisfaction

⚙️ Technologies & Libraries
Python

Pandas, NumPy, Matplotlib – data analysis & visualization

Scikit-learn – train/test split, model tuning, evaluation metrics

XGBoost – powerful gradient boosting classifier

📊 Dataset
The model uses Invistico_Airline.csv, which contains detailed passenger data, including service ratings and demographic information.
Key target: satisfaction (satisfied vs. dissatisfied).

✅ Steps
Import libraries

python
Zkopírovat
Upravit
import numpy as np
import pandas as pd
import matplotlib as plt
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

from xgboost import XGBClassifier, plot_importance
Load data

python
Zkopírovat
Upravit
airline_data = pd.read_csv('Invistico_Airline.csv', error_bad_lines=False)
Preprocess data (cleaning, encoding, splitting)

Train and tune the XGBoost classifier

Evaluate performance and compare with decision tree & random forest models

Analyze feature importance to identify key factors affecting satisfaction
