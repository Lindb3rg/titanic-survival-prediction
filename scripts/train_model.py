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


"""
Data exploration

"""

df_train.head()
df_train.count()
df_train.info()      
df_train.describe() 
df_train.isnull().sum()


""""
Preparing dataset

"""


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

train = df_train[features + ["Survived"]].copy()

# Filling Null values in Age with its median
train['Age'] = train['Age'].fillna(train['Age'].median())

# Filling Null values in Embark with the most common port
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

# Encoding Sex and Embarked into binary categories
train = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)

X_train = train.drop('Survived', axis=1)
y_train = train['Survived']

test = df_test[features].copy()

# Handling missing values using training data statistics
test['Age'] = test['Age'].fillna(df_train['Age'].median())        # Using original training median
test['Fare'] = test['Fare'].fillna(df_train['Fare'].median())    # Filling missing fare
test['Embarked'] = test['Embarked'].fillna(df_train['Embarked'].mode()[0])  # Use original training mode

# Applying same encoding as training data
test = pd.get_dummies(test, columns=['Sex', 'Embarked'], drop_first=True)

X_test = test

# Dropping PassengerId for referenece target
y_true = y_test.drop(columns="PassengerId")


"""
Instantiating Model

"""
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train,y_train)
predictions = model.predict(X_test)


"""
Extracting feature importances and visualizing it

"""


importances = model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names, 
    'Importance': importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance - Which factors most influence survival?')
plt.show()

"""
Calculating Scores

"""

# Accuracy Score (True Positives + True Negatives) / Total Predictions
accuracy_score = accuracy_score(predictions,y_true)
print(f"Accuracy score: {accuracy_score}.")

# Classification Score
cr = classification_report(predictions,y_true)
print(f"Classification report: \n {cr}")

"""
Save model to file

"""

joblib.dump(X_train.columns.tolist(), "rf_model_titanic_columns.joblib")
joblib.dump(model, 'rf_model_titanic.joblib')

print("Model saved successfully!")


