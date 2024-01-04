import streamlit as st
from pickleshare import PickleShareDB
import pandas as pd
import numpy as np
from pickleshare import PickleShareDB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def load_model_and_encoders():
    # Load the model from PickleShareDB
    db = PickleShareDB('model.pkl')
    model = db['model']

    # Load the label encoder from PickleShareDB
    db = PickleShareDB('label_encoder.pkl')
    label_encoder = db['label_encoder']

    return model, label_encoder

def preprocess_input(input_data, label_encoder):
    # Preprocess the input data and apply label encoding
    for col in input_data.select_dtypes(include='object').columns:
        input_data[col] = label_encoder.transform(input_data[col])
    return input_data

def main():
    st.title('Fraud Detection Web Page')
    st.write('Enter the details below to check if the claim is fraud or not.')

    # Load the model and label encoder
    model, label_encoder = load_model_and_encoders()
    policy_states=[]

    # Create input fields for user to enter data
    feature_names = ['incident_severity','insured_hobbies','vehicle_claim','total_claim_amount','property_claim','incident_state','injury_claim','insured_occupation','months_as_customer','policy_annual_premium']  # Replace with actual column names
    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.text_input(feature)

    if st.button('Predict'):
        # Convert user input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make predictions
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.write('Fraudulent claim detected!')
        else:
            st.write('Claim is not fraudulent.')

if __name__ == '__main__':
    main()
