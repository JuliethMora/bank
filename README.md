Bank Term Deposit Subscription Prediction
Project Overview
This project aims to predict whether a client will subscribe to a bank term deposit based on direct marketing campaign data. The primary goal is to develop a robust classification model that can help banking institutions optimize their marketing strategies by identifying potential subscribers more effectively.

Dataset
The dataset used for this project is bank-additional-full.csv, which contains information about direct marketing campaigns of a Portuguese banking institution. It includes client demographic data, attributes related to the last contact of the current campaign, social and economic context attributes, and the target variable 'y' (whether the client subscribed to a term deposit).

Data Preprocessing
Initial Inspection: The dataset was loaded and inspected for structure, data types, and missing values. It was found to contain 41,188 records with no missing values.
Target Variable Transformation: The target variable 'y' was converted from categorical ('yes', 'no') to numerical (1, 0).
Categorical Feature Encoding: All categorical variables (e.g., 'job', 'marital', 'education') were transformed into numerical representations using LabelEncoder.
Dataset Balancing: The original dataset exhibited a significant class imbalance in the target variable 'y'. To address this, random undersampling was applied to the majority class ('no' subscription) to create a balanced dataset, ensuring fair training for the models.
Data Splitting: The balanced dataset was split into training, validation, and test sets. A 70/30 split was used for initial train/test, and then the training portion was further split into training (80%) and validation (20%).
Feature Scaling: Numerical features were scaled using MinMaxScaler on the training data, and then the same scaler was applied to the validation and test sets to ensure consistent scaling.
Model Development and Evaluation
Two classification models were developed and evaluated:

1. Logistic Regression (LR)
An initial Logistic Regression model was trained and evaluated on the preprocessed data.
Accuracy Score: 0.8506
AUC Score: 0.9277
Classification Report: Showed balanced precision, recall, and F1-scores around 0.85 for both classes.
2. Gradient Boosting Classifier (GB)
A Gradient Boosting Classifier was trained and initially evaluated.
Accuracy Score: 0.8865
AUC Score: 0.9494
Hyperparameter Tuning for Gradient Boosting
GridSearchCV was employed to find optimal hyperparameters for the Gradient Boosting Classifier using roc_auc as the scoring metric.
Best Hyperparameters: {'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}.
Optimized GB Accuracy: 0.8883
Optimized GB AUC: 0.9479
Optimized GB Classification Report: Showed improved precision for class 0 (0.93) and recall for class 1 (0.94), leading to F1-scores of 0.88 and 0.89 respectively.
