
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, precision_score, recall_score, f1_score, auc
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="No-Show Predictor", layout="wide", page_icon="🏥")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏥 AI-Powered No-Show Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# =============================================
# CONFIGURE YOUR FILE PATHS HERE
# =============================================
MODEL_PATH = "C:\\Users\\krish\\OneDrive\\Desktop\\medical_no_show\\no_show_medical.pkl" 
DATA_PATH = "C:\\Users\\krish\\OneDrive\Desktop\\medical_no_show\\medical_final.csv"

# Load model automatically
@st.cache_resource
def load_model(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {path}")
        st.info("💡 Please update MODEL_PATH variable in the code with correct path")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.exception(e)
        st.stop()

# Load data automatically
@st.cache_data
def load_data(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.warning(f"⚠️ Data file not found: {path}")
        return None
    except Exception as e:
        st.warning(f"⚠️ Error loading CSV: {e}")
        return None

# Load the model
model = load_model(MODEL_PATH)

# Load CSV data if available
df = load_data(DATA_PATH)


# Extract unique cities from data
unique_cities = ["Other"]
if df is not None and 'city' in df.columns:
    try:
        cities = df['city'].dropna().astype(str).str.strip().str.lower().unique().tolist()
        unique_cities = sorted([c.title() for c in cities if c])
        unique_cities = ["Other"] + unique_cities
    except Exception:
        pass

# Navigation
page = st.sidebar.radio("📍 Navigate to", ["🔮 Live Prediction"])

# ---------------------------------
# FEATURE ENGINEERING FUNCTION
# ---------------------------------
def prepare_features(input_df):
    """Prepare features for prediction matching training pipeline"""
    X = input_df.copy()
    
    # Convert date columns
    date_cols = ['appointment_date', 'date_of_birth', 'entry_service_date']
    for col in date_cols:
        if col in X.columns:
            X[col] = pd.to_datetime(X[col], dayfirst=True, errors='coerce')
    
    # Create temporal features
    if 'appointment_date' in X.columns:
        X['appointment_month_num'] = X['appointment_date'].dt.month
        X['appointment_day_of_week'] = X['appointment_date'].dt.dayofweek
        X['appointment_year'] = X['appointment_date'].dt.year
    
    if 'date_of_birth' in X.columns:
        X['birth_year'] = X['date_of_birth'].dt.year.fillna(2011).astype(int)
    
    if 'entry_service_date' in X.columns:
        X['entry_service_year'] = X['entry_service_date'].dt.year.fillna(2016).astype(int)
        X['entry_service_month'] = X['entry_service_date'].dt.month.fillna(5).astype(int)
    
    # Calculate days in advance
    if 'appointment_date' in X.columns and 'entry_service_date' in X.columns:
        X['days_advance'] = (X['appointment_date'] - X['entry_service_date']).dt.days.fillna(269)
    
    # Drop original date columns
    X = X.drop(columns=date_cols, errors='ignore')
    
    # One-hot encode categorical variables
    cat_cols = ['specialty', 'gender', 'disability', 'city', 'appointment_shift', 
                'rain_intensity', 'heat_intensity']
    cat_cols = [c for c in cat_cols if c in X.columns]
    
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    
    # Align with model features
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
        # Add missing columns with 0
        for col in model_features:
            if col not in X.columns:
                X[col] = 0
        # Keep only model features in correct order
        X = X[model_features]
    
    return X

# ---------------------------------
# 1. LIVE PREDICTION PAGE
# ---------------------------------
if page == "🔮 Live Prediction":
    st.header("🔮 Live No-Show Risk Prediction")
    st.markdown("Enter patient appointment details below to predict no-show risk.")
    
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("👤 Patient Info")
            gender = st.selectbox("Gender", ['M', 'F'])
            age = st.slider("Age", 0, 120, 30)
            dob = st.date_input("Date of Birth", value=pd.to_datetime("1990-01-01"))
            companion = st.checkbox("Needs Companion")
            disability = st.selectbox("Disability", ['no', 'yes'])
        
        with col2:
            st.subheader("📅 Appointment Details")
            specialty = st.selectbox("Specialty", [
                'physiotherapy', 'psychotherapy', 'speech therapy',
                'occupational therapy', 'not given', 'pedagogo', 'enf'
            ])
            appt_date = st.date_input("Appointment Date")
            shift = st.selectbox("Shift", ['morning', 'afternoon'])
            entry_date = st.date_input("Entry Service Date", value=pd.to_datetime("2020-01-01"))
            city_input = st.selectbox("City", unique_cities)
        
        with col3:
            st.subheader("🌤️ Weather Conditions")
            avg_temp = st.number_input("Avg Temperature (°C)", value=20.0, step=0.5)
            max_temp = st.number_input("Max Temperature (°C)", value=25.0, step=0.5)
            avg_rain = st.number_input("Avg Rain (mm)", value=0.0, step=0.1)
            max_rain = st.number_input("Max Rain (mm)", value=0.0, step=0.1)
            rainy_yest = st.checkbox("Rained Yesterday")
            storm_yest = st.checkbox("Storm Yesterday")
            rain_int = st.selectbox("Rain Intensity", ['no rain', 'weak', 'moderate', 'heavy'])
            heat_int = st.selectbox("Heat Intensity", ['cold', 'mild', 'warm', 'heavy warm'])
        
        submit = st.form_submit_button("🎯 Predict Risk", use_container_width=True)
        
        if submit:
            try:
                # Prepare input data
                city = city_input.lower() if city_input != "Other" else "other"
                disability_val = 1 if disability == 'yes' else 0
                
                input_data = {
                    'specialty': [specialty],
                    'gender': [gender],
                    'city': [city],
                    'appointment_shift': [shift],
                    'age': [age],
                    'under_12_years_old': [1 if age < 12 else 0],
                    'over_60_years_old': [1 if age > 60 else 0],
                    'patient_needs_companion': [1 if companion else 0],
                    'disability': [disability_val],
                    'average_temp_day': [avg_temp],
                    'average_rain_day': [avg_rain],
                    'max_temp_day': [max_temp],
                    'max_rain_day': [max_rain],
                    'rainy_day_before': [1 if rainy_yest else 0],
                    'storm_day_before': [1 if storm_yest else 0],
                    'rain_intensity': [rain_int],
                    'heat_intensity': [heat_int],
                    'appointment_date': [appt_date.strftime('%d/%m/%Y')],
                    'date_of_birth': [dob.strftime('%d/%m/%Y')],
                    'entry_service_date': [entry_date.strftime('%d/%m/%Y')]
                }
                
                input_df = pd.DataFrame(input_data)
                
                # Prepare features
                X_processed = prepare_features(input_df)
                
                # Make prediction
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_processed)[0][1]
                elif hasattr(model, 'decision_function'):
                    score = model.decision_function(X_processed)[0]
                    prob = 1.0 / (1.0 + np.exp(-score))
                else:
                    st.error("❌ Model doesn't support probability predictions")
                    st.stop()
                
                pred = int(prob > 0.5)
                
                # Determine risk level
                if prob < 0.3:
                    risk = "LOW"
                    color = "#00C853"
                elif prob < 0.7:
                    risk = "MEDIUM"
                    color = "#FF9800"
                else:
                    risk = "HIGH"
                    color = "#F44336"
                
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Display results
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("🎯 Prediction", "No-Show ❌" if pred else "Will Show ✅")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_b:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("⚠️ Risk Level", risk)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_c:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("📈 No-Show Probability", f"{prob:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={'text': f"<b>Risk Level: {risk}</b>", 'font': {'size': 24, 'color': color}},
                    number={'suffix': "%", 'font': {'size': 40}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 2},
                        'bar': {'color': color, 'thickness': 0.75},
                        'steps': [
                            {'range': [0, 30], 'color': '#E8F5E9'},
                            {'range': [30, 70], 'color': '#FFF3E0'},
                            {'range': [70, 100], 'color': '#FFEBEE'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if risk == "HIGH":
                    st.error("🚨 **High Risk**: Consider sending reminder SMS/email and confirming appointment")
                elif risk == "MEDIUM":
                    st.warning("⚠️ **Medium Risk**: Send reminder notification before appointment")
                else:
                    st.success("✅ **Low Risk**: Standard appointment protocol")
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.exception(e)

