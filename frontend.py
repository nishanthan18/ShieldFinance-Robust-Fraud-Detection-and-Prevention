import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from predict import preprocess_data, predict_fraud

# Set the page layout and title (this should be the first Streamlit command)
st.set_page_config(page_title="Financial Fraud Detection", layout="centered")

# Custom CSS for styling titles
st.markdown("""
    <style>
    body {
        color: #000000;  /* Default text color */
    }
    .stApp {
        color: #000000;
    }
    .css-1d391kg {
        color: #000000;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .css-1aumxhk {
        color: #000000;
    }
    .css-1cpxqw2 {
        color: #000000;
    }
    .css-h5rgaw, .css-1v0mbdj, .css-1r6slb0, .css-1uyte9r {
        color: #000000; /* Black color for all titles */
        text-align: center; /* Center-align the titles */
    }
    </style>
    """, unsafe_allow_html=True)

# Paths to your model and scaler
model_path = "fraud_detection_model_final.keras"
scaler_path = "scaler.pkl"

# Load the model and scaler
model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Title and instructions
st.markdown("<h1 style='text-align: center;'>💳 Financial Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("""
Welcome to the Financial Fraud Detection System. Use this tool to check transactions for potential fraud.
""")

# Sidebar for navigation
st.sidebar.header("Navigation")
st.sidebar.markdown("""
- [Home](#home)
- [Submit Transaction](#submit-transaction)
- [View CSV File](#view-csv-file)
""")

# Form for user input
st.header("Submit Transaction Details")

with st.form(key='transaction_form'):
    acc_no = st.text_input("Account No", max_chars=24)
    tran_date = st.text_input("Transaction Date (YYYY-MM-DD)", max_chars=10)
    tran_detail = st.text_input("Transaction Details", max_chars=15)
    chq_no = st.number_input("CHQ.NO.")
    val_date = st.text_input("VALUE DATE", value=tran_date)
    withdrawal_amt = st.number_input("WITHDRAWAL AMT", min_value=0.0, step=0.01)
    deposit_amt = st.number_input("DEPOSIT AMT", min_value=0.0, step=0.01)
    balance_amt = st.number_input("BALANCE AMT", min_value=0.0, step=0.01)

    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    data = {
        'Account No': acc_no,
        'DATE': tran_date,
        'TRANSACTION DETAILS': tran_detail,
        'CHQ.NO.': chq_no,
        'VALUE DATE': val_date,
        'WITHDRAWAL AMT': withdrawal_amt,
        'DEPOSIT AMT': deposit_amt,
        'BALANCE AMT': balance_amt
    }

    df = pd.DataFrame([data])
    processed_data = preprocess_data(df)

    fraud_probability = predict_fraud(model, processed_data, scaler)

    df['Fraud Probability'] = fraud_probability

    df.to_csv("transaction_data_with_fraud_prob.csv", index=False)

    if any(fraud_probability == 1):
        st.error("🚨 Alert: Transaction(s) with fraud probability detected. Admin action required.")
    else:
        st.success("✅ Transaction completed successfully! Check the CSV file for details.")

# Section to display the CSV file
st.header("View CSV File")

if st.button("Show CSV File"):
    try:
        csv_data = pd.read_csv("transaction_data_with_fraud_prob.csv")
        st.write("CSV file contents:")
        st.dataframe(csv_data)
    except FileNotFoundError:
        st.error("CSV file not found. Please submit data first.")

# Additional footer info or links
st.sidebar.markdown("#### About")
st.sidebar.info("This application helps in detecting potential fraudulent transactions in financial data using machine learning models.")
