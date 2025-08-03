#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

#Load the Dataset
df = pd.read_csv("insurance.csv")
df.head()

#Understand the Dataset
df.info()
df.describe()
df.isnull().sum()
df['region'].value_counts()
df['smoker'].value_counts()

#Exploratory Data Analysis (EDA)
sns.pairplot(df, hue='smoker')
sns.boxplot(x='smoker', y='charges', data=df)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

#Alternatively, use this code for bypassing value error 
#sns.pairplot(df, hue='smoker')
#sns.boxplot(x='smoker', y='charges', data=df)
#sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')

#Data Preprocessing
# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Check new columns
df_encoded.head()

#Split the Data
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Build the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)



#Multiple Model Evaluation
from sklearn.ensemble import GradientBoostingRegressor

# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("🔹 Linear Regression")
print("  R² Score:", r2_score(y_test, y_pred_lr))
print("  RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print()

# 2. Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

print("🔹 Gradient Boosting Regressor")
print("  R² Score:", r2_score(y_test, y_pred_gb))
print("  RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_gb)))


#Make a Prediction Example
example = pd.DataFrame({
    'age': [40],
    'bmi': [28],
    'children': [2],
    'sex_male': [1],
    'smoker_yes': [0],
    'region_northwest': [0],
    'region_southeast': [1],
    'region_southwest': [0]
})

predicted_charge = model.predict(example)
print("Predicted Insurance Charge:", predicted_charge[0])
