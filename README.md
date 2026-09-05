# Heart Disease Risk Prediction

A machine learning based web application that predicts the risk of heart disease from clinical and physiological patient parameters.

The application is developed using Python, Scikit-learn, and Streamlit. It provides an interactive interface where users can enter patient information and receive a predicted heart disease risk along with the corresponding probability.

---

## Project Overview

Heart disease is one of the major health concerns worldwide. Early identification of potential risk can support further medical evaluation and awareness.

This project applies supervised machine learning techniques to analyze patient-related features and predict whether the input data indicates a higher likelihood of heart disease.

The system includes:

- Data preprocessing
- Exploratory data analysis
- Multiple machine learning models
- Model comparison
- Best-model selection
- Probability-based prediction
- Interactive Streamlit interface
- Prediction history
- Model performance information
- Professional PDF report generation
- Feature information and input guidance

---

## Key Features

### 1. Interactive Risk Prediction

Users can enter patient information through an easy-to-use Streamlit interface.

The application accepts the following 11 features:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Angina
- Oldpeak
- ST Slope

After entering the information, the application processes the data and generates a prediction.

---

### 2. Probability-Based Prediction

The application displays the predicted probability associated with the heart-disease class.

The probability is also presented using three application-defined risk categories:

| Probability | Risk Category |
|---|---|
| Less than 30% | Low Risk |
| 30% to less than 60% | Moderate Risk |
| 60% or higher | High Risk |

These categories are presentation categories defined for this application and should not be interpreted as clinical diagnostic thresholds.

---

### 3. Patient Summary

After prediction, the application displays the entered patient information in a structured table.

This allows users to review the input values before interpreting the prediction.

---

### 4. Prediction History

The application maintains prediction results during the current Streamlit session.

Users can review previously generated predictions without manually recording each result.

---

### 5. Professional PDF Report

The application can generate a professional PDF report containing:

- Assignment ID
- Date and time
- Patient information
- Prediction result
- Heart disease probability
- Model information
- Model performance
- Disclaimer
- Developer information

The report can be downloaded directly from the application.

---

### 6. Model Performance

The application includes a dedicated model performance section showing the evaluation results of the tested machine learning algorithms.

The evaluated models are:

- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Logistic Regression
- Decision Tree

---

## Machine Learning Models

Five supervised classification algorithms were evaluated.

### Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction performance and reduce overfitting.

### Support Vector Machine

Support Vector Machine finds an optimal decision boundary between different classes and can perform effectively on structured classification problems.

### K-Nearest Neighbors

K-Nearest Neighbors predicts a class based on the characteristics of nearby observations in the feature space.

### Logistic Regression

Logistic Regression is a statistical classification algorithm that estimates the probability of a binary outcome.

### Decision Tree

Decision Tree performs classification by creating a sequence of decision rules based on feature values.

---

## Model Evaluation

The models were evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 90.22% | 89.62% | 93.14% | 91.35% | 93.31% |
| SVM | 89.67% | 88.79% | 93.14% | 90.91% | 94.94% |
| KNN | 89.67% | 91.09% | 90.20% | 90.64% | 94.31% |
| Logistic Regression | 88.59% | 87.16% | 93.14% | 90.05% | 92.99% |
| Decision Tree | 79.35% | 81.37% | 81.37% | 81.37% | 79.10% |

### Selected Model

The Random Forest model was selected as the final prediction model because it achieved the highest F1-Score among the evaluated models.

**Random Forest F1-Score: 91.35%**

The model also achieved:

- Accuracy: 90.22%
- Precision: 89.62%
- Recall: 93.14%
- ROC-AUC: 93.31%

---

## Dataset

The project uses a heart disease dataset containing clinical and physiological patient attributes.

### Dataset Features

| Feature | Description |
|---|---|
| Age | Patient age in years |
| Sex | M = Male, F = Female |
| ChestPainType | Type of chest pain |
| RestingBP | Resting systolic blood pressure |
| Cholesterol | Serum cholesterol level |
| FastingBS | Fasting blood sugar category |
| RestingECG | Resting electrocardiogram result |
| MaxHR | Maximum heart rate |
| ExerciseAngina | Exercise-induced angina |
| Oldpeak | ST-segment depression |
| ST_Slope | Slope of the peak exercise ST segment |
| HeartDisease | Target variable |

---

## Feature Encoding

Some features use categorical codes.

### Sex

| Code | Meaning |
|---|---|
| M | Male |
| F | Female |

### ChestPainType

| Code | Meaning |
|---|---|
| ATA | Atypical Angina |
| NAP | Non-Anginal Pain |
| ASY | Asymptomatic |
| TA | Typical Angina |

### FastingBS

| Value | Meaning |
|---|---|
| 0 | Fasting blood sugar <= 120 mg/dL |
| 1 | Fasting blood sugar > 120 mg/dL |

### RestingECG

| Code | Meaning |
|---|---|
| Normal | Normal ECG |
| ST | ST-T wave abnormality |
| LVH | Left Ventricular Hypertrophy |

### ExerciseAngina

| Code | Meaning |
|---|---|
| N | No |
| Y | Yes |

### ST_Slope

| Code | Meaning |
|---|---|
| Up | Upsloping |
| Flat | Flat |
| Down | Downsloping |

---

## Data Preprocessing

The project uses a preprocessing pipeline to prepare the input data before model prediction.

### Numerical Features

Numerical features are standardized using:

`StandardScaler`

### Categorical Features

Categorical features are transformed using:

`OneHotEncoder`

The encoder uses:

`handle_unknown="ignore"`

This allows the application to safely process categorical values that were not encountered during model training.

---

## Model Pipeline

The trained preprocessing and prediction process is stored together in:

Live App: https://heart-disease-risk-prediction-jevin.streamlit.app

Developed by Jevin Kanani  
Data Science | Machine Learning

```text
