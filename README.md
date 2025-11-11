# HealthPlan Pro AI — Medical Insurance Cost Prediction (India 2025)

## Project Overview
HealthPlan Pro AI is an end-to-end machine learning application designed to predict personalized medical insurance costs for individuals based on demographic, health, and lifestyle factors.  
The system integrates data preprocessing, model training, and a RESTful backend API, connected to a modern web frontend built using Lovable.ai.  

This project demonstrates how predictive analytics and AI can be applied to the healthcare and insurance industry to enhance decision-making and improve customer transparency.

---

## Objectives
- Predict the expected medical insurance premium for users based on features such as age, BMI, smoking status, number of dependents, and region.  
- Analyze the relationship between personal health indicators and insurance cost.  
- Provide a real-time web application where users can enter data and instantly view their predicted insurance cost.

---

## Key Features
- End-to-end ML pipeline with feature preprocessing and model training.
- Custom-engineered features to enhance prediction accuracy:
  - `smoker_flag`
  - `senior_citizen_flag`
  - `bmi_smoker_interaction`
- Real-time API endpoint for insurance cost prediction using Flask.
- Clean integration with Lovable.ai frontend for an interactive user experience.
- Modular architecture allowing retraining, scaling, and deployment.

---

## Technical Summary
| Component | Description |
|------------|-------------|
| **Programming Language** | Python |
| **Frameworks** | Flask, Scikit-learn, XGBoost |
| **Libraries** | Pandas, NumPy, Joblib |
| **Frontend** | Lovable.ai (connected via REST API) |
| **Model Type** | XGBoost Regressor |
| **Evaluation Metrics** | R² Score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) |
| **Dataset** | Custom-engineered 2025 dataset simulating Indian health insurance data |


## System Architecture
1. **Data Layer:**  
   A structured dataset containing user demographic and lifestyle attributes along with insurance cost data.  

2. **Modeling Layer:**  
   Preprocessing with `ColumnTransformer` and `StandardScaler`, followed by an XGBoost regression model.  

3. **API Layer:**  
   A Flask backend exposing `/predict` endpoint for real-time predictions.  

4. **Frontend Layer:**  
   A Lovable.ai-based user interface that collects user inputs and fetches prediction results from the Flask backend.


## Project Workflow
1. Load and preprocess dataset (`medical_insurance_engineered.csv`).
2. Feature engineering and model training using XGBoost.
3. Evaluate model using R², MAE, and RMSE.
4. Save the trained model as `insurance_model.pkl`.
5. Build and host a Flask API to serve predictions.
6. Connect the backend API to a Lovable.ai frontend for user interaction.

---

## API Usage
**Endpoint:**  
