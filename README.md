
# 🏥 AI_Powered_Medical_Appointment_No_Show_Prediction

The project leverages Machine Learning and Streamlit to predict whether or not patients will show up for their medical appointments. It aims to reduce no-shows, optimize scheduling, and improve patient engagement for healthcare providers.

streamlit app : [link](https://medicalnoshowpy-7favft8ejlknxrnebvheqk.streamlit.app/)

Streamlit Dashboard : [link](https://dashboardpy-7.streamlit.app/)

---
## 🚀 Important Features

Data cleaning, preprocessing, training, assessment, and deployment comprise the end-to-end machine learning pipeline.

### Streamlit Interactive Web App:

 - Real-time no-show risk prediction.
 - dashboard containing insights and performance metrics.
 - Visualizations of model comparisons.

 ### Explicit Insights:

- Analysis of feature importance.
- identification of high-risk patients.
-  Visual performance summaries (ROC, confusion matrix, etc.).

---
## 🧩 Tech Stack

| Category         | Tools / Libraries               |
| ---------------- | ------------------------------- |
| Programming      | Python                          |
| Framework        | Streamlit                       |
| Data Processing  | Pandas, NumPy                   |
| Machine Learning | Scikit-learn, XGBoost, CatBoost |
| Visualization    | Plotly, Matplotlib, Seaborn     |
| Model Storage    | Pickle (.pkl)                   |


----

## 🧠 Workflow

### 1️⃣ Data Preprocessing (cleaning_no_show.ipynb)

- Handled missing values, duplicates, and outliers.
- Performed encoding and normalization.
- Feature engineering: created columns like days_advance, appointment_day_of_week, etc.

### 2️⃣ Model Training (no_show.ipynb)

Compared several models:

 - Gradient Boosting
 - Random Forest
 - XGBoost
 - CatBoost
 - Logistic Regression and more
- Achieved 90.3% accuracy and AUC = 0.7645 with Gradient Boosting as the best model.

### 3️⃣ Deployment (medical_no_show.py)

 Built a Streamlit dashboard with:
 - Live Prediction form to input patient data.
 - Dashboard showing metrics, confusion matrix, ROC, and feature importance.
 - Model Performance tab comparing multiple algorithms.
