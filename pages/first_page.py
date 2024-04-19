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
        # Filter rows where 'company' column equals 'company_name'
        filtered_df = df[df['company'] == company]
        # Extract the 'name' column from the filtered DataFrame
        models = filtered_df['name'].tolist()
        model = st.selectbox("Model", models)
        year = st.selectbox("Year of Purchase",years)
        year = int(year)
        fuel_type = st.selectbox("Fuel Type", fuel_types)
        kms_driven = st.number_input("Approximate Number of Kilometers Driven", min_value=0,)
        kms_driven = int(kms_driven)
        
    if not company or not model or not year or not fuel_type or not kms_driven:
        st.warning("Please enter all car details to continue")


    if st.sidebar.button("Submit"):
        question = f"Car Details: {company}, {model}, Year: {year}, Fuel Type: {fuel_type}, KMs Driven: {kms_driven}"
        # Prepare data for prediction
        data = pd.DataFrame({'name': [model], 'company': [company], 'year': [year], 'kms_driven': [kms_driven], 'fuel_type': [fuel_type]})
        # Predict
        prediction = pipe.predict(data)
        st.write("Prediction:", prediction)

if __name__ == '__main__':
    main()
