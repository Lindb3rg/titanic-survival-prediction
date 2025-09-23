from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,\
    classification_report

import joblib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')
y_test = pd.read_csv("../data/gender_submission.csv")


df_train.head()
df_train.count()
df_train.info()      
df_train.describe() 
df_train.isnull().sum()