import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Load the saved model
# -------------------------------
with open("loan_approval_model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS styling
# -------------------------------
st.markdown("""
    <style>
    body {
        background-color: #0d0d0d;
        color: white;
    }
    .stApp {
        background: linear-gradient(145deg, #000000, #1a1a1a);
        color: white;
    }
    .stTextInput, .stNumberInput, .stSelectbox {
        background-color: #262626 !important;
        color: white !important;
    }
    .css-1d391kg, .css-1v0mbdj {
        background-color: #1a1a1a !important;
    }
    .stButton>button {
        background-color: #0078ff;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #005ce6;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar section
# -------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
st.sidebar.title("👤 Developer Info")
st.sidebar.markdown("""
**Name:** Dev Landkar  
**Role:** Data Analyst / ML Developer  
**GitHub:** [@Shubhamlandkar](https://github.com/Shubhamlandkar)  
**LinkedIn:** [Shubh Landkar](https://www.linkedin.com)  
""")

st.sidebar.write("📞 Contact: Shubhlandkar@email.com")

# -------------------------------
# App header
# -------------------------------
st.title("💰 Loan Approval Prediction System")
st.markdown("### Predict whether a loan will be approved based on applicant details")

# -------------------------------
# Input section
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.number_input("Number of Dependents", 0, 10, 0)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Income", 0, 100000, 5000)
    coapplicant_income = st.number_input("Coapplicant Income", 0, 50000, 0)
    loan_amount = st.number_input("Loan Amount (in thousands)", 0, 1000, 100)
    loan_amount_term = st.number_input("Loan Amount Term (in days)", 0, 500, 360)
    credit_history = st.selectbox("Credit History", [0.0, 1.0])

property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -------------------------------
# Convert inputs to numeric for model
# -------------------------------
def preprocess_input():
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    education_val = 1 if education == "Graduate" else 0
    self_employed_val = 1 if self_employed == "Yes" else 0
    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_val = property_map[property_area]
    
    return np.array([[gender_val, married_val, dependents, education_val,
                      self_employed_val, applicant_income, coapplicant_income,
                      loan_amount, loan_amount_term, credit_history, property_val]])

# -------------------------------
# Prediction button
# -------------------------------
if st.button("🔍 Predict Loan Approval"):
    input_data = preprocess_input()
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Congratulations! Your Loan is **Approved**.")
    else:
        st.error("❌ Sorry, your Loan **is not Approved**.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("💼 *Developed by Shubh Landkar | Machine Learning Project*")
