# 🛒 Social Network Ads Purchase Prediction

## Link to Social Network Ads Purchase Prediction App
<a href= "https://socialnetworkadv-prediction-application-jpxcxpbgegihsdlshha7aa.streamlit.app/" view>To access the app</a>

## Table of Content
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Materials and Method](#materials-and-method)
- [Results and DisCcussion](#results-and-discussion)
-  [Conclusion](#conclusion)


## Abstract
This project presents a machine learning-based web application that predicts whether users are likely to purchase a product based on their social network advertisement data. Leveraging the power of the XGBoost algorithm, SMOTE for handling class imbalance, and data preprocessing techniques, the solution provides both single and batch predictions via an interactive Streamlit user interface. The model was trained and evaluated using a dataset containing user demographics and purchase outcomes. The web app is designed to support both individual predictions and bulk analysis from CSV uploads, enabling wide usability across business and academic contexts.


## Introduction

With the rise of targeted marketing on social media platforms, understanding which users are most likely to make purchases has become a strategic advantage. Traditional approaches often rely on heuristic or rule-based methods, which lack adaptability and precision. In this project, we aim to provide a predictive solution using supervised machine learning that forecasts user behavior based on demographic features: Age, Estimated Salary, and Gender. The prediction model is deployed as a web application using Streamlit, making it accessible and user-friendly for marketers and analysts.


## Materials and Method

#### Dataset

- The dataset used is the classic "Social Network Ads" dataset. It contains:

- User ID: unique identifier (dropped during processing)

- Gender: Male or Female

- Age: integer

- EstimatedSalary: numerical value

- Purchased: binary target variable indicating whether the user made a purchase (0 = No, 1 = Yes)

#### Preprocessing Steps

- The User ID column was dropped as it does not contribute to prediction.

- The Gender column was encoded: Male = 1, Female = 0

- Features (Age, EstimatedSalary, Gender) were standardized using StandardScaler.

#### Addressing Class Imbalance

The dataset showed imbalance in the Purchased classes. To address this, SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the classes before model training.

#### Model Training

- Algorithm Used: XGBoost Classifier

- Parameters: use_label_encoder=False, eval_metric='logloss', random_state=42

- Train-Test Split: 80% training, 20% testing after SMOTE

#### Evaluation Metrics

- Confusion Matrix

- Classification Report (precision, recall, f1-score)

- ROC-AUC Score

#### Deployment

A web interface was created using Streamlit

The app provides:

- Single Prediction: manual input of user details

- Batch Prediction: upload a CSV file with multiple users

- The model (xgb_smote_model.joblib) and scaler (scaler.joblib) were serialized using joblib

## Results and Discussion

1. Model Performance

After training and evaluating the model, the results were:

- Accuracy: 88% (based on classification report)

- Precision and Recall: High for both classes, especially after SMOTE balancing

- ROC-AUC Score: Above 0.9, indicating strong classification capability

These results demonstrate that the XGBoost model effectively identifies potential buyers based on age, salary, and gender. SMOTE significantly improved the model's ability to recognize minority class instances, preventing biased predictions.

2. Web App Functionality

- Ease of Use: Users can enter inputs or upload files without technical background

- Responsiveness: Fast and efficient prediction using pre-trained model

- Error Handling: Errors such as missing columns, incorrect formats, or invalid inputs are managed gracefully

## Conclusion

The Social Network Ads Purchase Prediction system successfully integrates machine learning with an intuitive user interface to solve a real-world classification problem. The model achieves high accuracy through data preprocessing, class balancing, and use of the XGBoost algorithm. The Streamlit app broadens accessibility, allowing both marketers and data practitioners to use predictive insights to enhance targeted advertising strategies.

This project demonstrates the importance of combining sound machine learning practices with user-friendly deployment platforms for impactful data science solutions.

