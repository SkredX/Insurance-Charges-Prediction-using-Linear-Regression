# Insurance Charges Prediction

This project aims to predict individual medical insurance charges using machine learning models. Built using Python in a Jupyter Notebook environment, the model is trained on the publicly available Medical Cost Personal Dataset from Kaggle. The objective is to understand the relationship between personal attributes (such as age, BMI, smoker status) and medical expenses.

## Table of Contents

- [Problem Statement](#problem-statement)  
- [Dataset Overview](#dataset-overview)  
- [Approach](#approach)  
- [Technologies Used](#technologies-used)  
- [Model Evaluation](#model-evaluation)  
- [Key Learnings](#key-learnings)  
- [Conclusion](#conclusion)

## Problem Statement

Health insurance providers often need to estimate insurance charges for individuals based on personal and lifestyle factors. This project uses regression models to build a predictive solution that estimates insurance costs, helping both providers and individuals make informed financial and health-related decisions.

## Dataset Overview

- **Source**: [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/mirichoi0218/insurance)  
- **Size**: 1338 rows × 7 columns  
- **Features**:
  - `age`: Age of the primary beneficiary
  - `sex`: Gender
  - `bmi`: Body Mass Index
  - `children`: Number of dependents
  - `smoker`: Smoking status
  - `region`: Residential area in the US
  - `charges`: Individual medical costs billed by health insurance (Target Variable)

## Approach

1. **Data Loading & Exploration**: Initial inspection of data types, null values, and distribution.
2. **Data Preprocessing**:
   - Encoding categorical variables using one-hot encoding
   - Splitting data into training and testing sets
3. **Model Building**:
   - Baseline model: Linear Regression
   - Advanced model: Gradient Boosting Regressor for comparison
4. **Model Evaluation**:
   - Performance measured using R² Score and RMSE
   - Visual comparison of predictions vs actual values

## Technologies Used

- **Language**: Python  
- **Environment**: Jupyter Notebook (Anaconda)  
- **Libraries**:
  - `pandas`, `numpy` for data handling
  - `matplotlib`, `seaborn` for visualization
  - `scikit-learn` for modeling and evaluation

## Model Evaluation

| Model                     | R² Score | RMSE         |
|---------------------------|----------|--------------|
| Linear Regression         | ~0.75    | ~6000–7000   |
| Gradient Boosting Regressor | ~0.88 | ~4000–5000   |

The Gradient Boosting model significantly improves accuracy by capturing non-linear relationships in the data.

## Key Learnings

- Linear Regression provides a clear and interpretable baseline for continuous prediction tasks.
- Categorical data must be transformed using one-hot encoding before model training.
- Smoker status and BMI are among the most influential features in predicting insurance charges.
- Ensemble models like Gradient Boosting outperform simple linear models when data relationships are non-linear.

## Conclusion

This project demonstrates how regression models can be effectively applied to real-world problems like insurance cost prediction. While Linear Regression offers interpretability, Gradient Boosting delivers better predictive accuracy. The model pipeline developed here can be extended to similar applications in finance, healthcare, or risk assessment domains.