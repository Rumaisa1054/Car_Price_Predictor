import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(layout="wide")


pipe = pickle.load(open('LinearRegressionModel.pkl','rb'))

st.title("Car Price Prediction App")

def main():
    with st.sidebar:
        st.subheader("Enter Car Details")
        df = pd.read_csv("Cleaned_Car_data.csv")
        companies = df['company']
        
        years = df['year']
        fuel_types = df['fuel_type']
        kms_drivens = df['kms_driven']
        company = st.selectbox("Company", companies)
        company = str(company)
        # Filter rows where 'company' column equals 'company_name'
        filtered_df = df[df['company'] == company]
        # Extract the 'name' column from the filtered DataFrame
        models = filtered_df['name'].tolist()
        model = st.selectbox("Model", models)
        model = str(model)
        year = st.selectbox("Year of Purchase",years)
        year = int(year)
        fuel_type = st.selectbox("Fuel Type", fuel_types)
        fuel_type = str(fuel_type)
        kms_driven = st.number_input("Approximate Number of Kilometers Driven", min_value=0,)
        kms_driven = int(kms_driven)
        
    if not company or not model or not year or not fuel_type or not kms_driven:
        st.warning("Please enter all car details to continue")


    if st.sidebar.button("Submit"):
        # Prepare input data
        input_data = pd.DataFrame({'name': [model], 'company': [company], 'year': [year], 'kms_driven': [kms_driven], 'fuel_type': [fuel_type]})
        # One-hot encode categorical variables
        encoded_data = pd.get_dummies(input_data)
        # Make sure the columns in the encoded data match the columns used during training
        missing_cols = set(pipe.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()) - set(encoded_data.columns)
        for col in missing_cols:
            encoded_data[col] = 0
        encoded_data = encoded_data[pipe.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()]
        # Predict
        prediction = pipe.predict(encoded_data)
        st.write("Prediction:", prediction)
if __name__ == '__main__':
    main()
