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
        # Encoding categorical variables
        # Your encoding code here...
        # Make sure to use the same encoding technique used during training
        # Predict
        prediction = pipe.predict(input_data)
        st.write("Prediction:", prediction)
if __name__ == '__main__':
    main()
