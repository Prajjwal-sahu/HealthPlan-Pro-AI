import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Load dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("medical_insurance_engineered.csv")
    return df

df = load_data()

st.title("🏥 Medical Insurance Cost Prediction Dashboard (India 2025)")
st.write("Compare insurance charges by company, predict costs, and analyze pricing patterns.")

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔍 Filter Data")

region = st.sidebar.multiselect("Select Region", df['region'].unique(), default=df['region'].unique())
company = st.sidebar.multiselect("Select Company", df['company_name'].unique(), default=df['company_name'].unique())
smoker = st.sidebar.multiselect("Smoker Status", df['smoker'].unique(), default=df['smoker'].unique())
budget = st.sidebar.multiselect("Budget Category", df['budget_category'].unique(), default=df['budget_category'].unique())

filtered_df = df[
    df['region'].isin(region) &
    df['company_name'].isin(company) &
    df['smoker'].isin(smoker) &
    df['budget_category'].isin(budget)
]

st.write(f"### 📊 Showing {filtered_df.shape[0]} Records After Filters")

# ----------------------------
# Company-wise Comparison
# ----------------------------
st.subheader("🏢 Company-wise Average Charges")

company_avg = filtered_df.groupby('company_name')['charges'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=company_avg.index, y=company_avg.values, palette='coolwarm')
plt.xticks(rotation=30)
plt.ylabel("Average Charges (₹)")
plt.title("Average Charges per Company")
st.pyplot(fig)

# ----------------------------
# Company vs Smoker Comparison
# ----------------------------
st.subheader("💨 Charges by Company and Smoker Status")
fig, ax = plt.subplots(figsize=(9,4))
sns.boxplot(data=filtered_df, x='company_name', y='charges', hue='smoker', palette='Set2')
plt.xticks(rotation=30)
plt.title("Charges by Company and Smoker Status")
st.pyplot(fig)

# ----------------------------
# Model Training Section
# ----------------------------
st.subheader("🧠 Train Prediction Model")

X = df.drop(columns=['charges', 'log_charges'])
y = df['charges']

categorical_cols = ['sex', 'smoker', 'region', 'company_name']
numerical_cols = ['age', 'bmi', 'children', 'smoker_flag', 'senior_citizen_flag', 'bmi_smoker_interaction']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

model_choice = st.selectbox("Choose Model", ["Random Forest", "XGBoost"])
if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=200, random_state=42)
else:
    model = XGBRegressor(n_estimators=250, learning_rate=0.1, max_depth=5, random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write(f"**Model Performance ({model_choice}):**")
st.metric("R² Score", f"{r2:.3f}")
st.metric("MAE (₹)", f"{mae:.2f}")
st.metric("RMSE (₹)", f"{rmse:.2f}")

# ----------------------------
# Prediction Form
# ----------------------------
st.subheader("📈 Predict Your Insurance Cost")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", 18, 80, 30)
    sex = st.selectbox("Sex", ["male", "female"])
with col2:
    bmi = st.number_input("BMI", 15.0, 45.0, 25.0)
    children = st.number_input("Children", 0, 5, 1)
with col3:
    smoker_input = st.selectbox("Smoker", ["yes", "no"])
    region_input = st.selectbox("Region", df['region'].unique())
company_input = st.selectbox("Company", df['company_name'].unique())

# Derived features
smoker_flag = 1 if smoker_input == "yes" else 0
senior_citizen_flag = 1 if age >= 60 else 0
bmi_smoker_interaction = bmi * smoker_flag

input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker_input],
    'region': [region_input],
    'company_name': [company_input],
    'smoker_flag': [smoker_flag],
    'senior_citizen_flag': [senior_citizen_flag],
    'bmi_smoker_interaction': [bmi_smoker_interaction]
})

if st.button("Predict Insurance Cost"):
    prediction = pipeline.predict(input_data)[0]
    st.success(f"💰 Predicted Insurance Cost: ₹{prediction:,.2f}")

# ----------------------------
# Company Comparison (side by side)
# ----------------------------
st.subheader("🏆 Compare Companies Side-by-Side")

compare_df = df.groupby('company_name')[['charges', 'age', 'bmi']].mean().reset_index()
st.dataframe(compare_df.style.format({'charges': '₹{:.2f}', 'bmi': '{:.1f}', 'age': '{:.0f}'}))
