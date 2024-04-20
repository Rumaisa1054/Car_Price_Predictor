import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(layout="wide")
# Load the trained model
with open('LinearRegressionModel.pkl', 'rb') as model_file:
    pipe = pickle.load(model_file)
    
st.title("Car Price Prediction App")

def main():
    try:
        # Load the cleaned DataFrame
        df = pd.read_csv("Cleaned_Car_data.csv")

        with st.sidebar:
            st.subheader("Enter Car Details")
            companies = df['company'].unique()
            years = df['year'].unique()
            fuel_types = df['fuel_type'].unique()
            kms_drivens = df['kms_driven'].unique()
            
            company = st.selectbox("Company", companies)
            company = str(company)
            
            # Filter rows where 'company' column equals 'company_name'
            filtered_df = df[df['company'] == company]
            models = filtered_df['name'].tolist()
            model = st.selectbox("Model", models)
            model = str(model)
            
            year = st.selectbox("Year of Purchase", years)
            year = int(year)
            
            fuel_type = st.selectbox("Fuel Type", fuel_types)
            fuel_type = str(fuel_type)
            
            kms_driven = st.number_input("Approximate Number of Kilometers Driven", min_value=0)
            kms_driven = int(kms_driven)
            
        if not company or not model or not year or not fuel_type or not kms_driven:
            st.warning("Please enter all car details to continue")

        if st.sidebar.button("Submit"):
            # Prepare input data
            input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                       data=[[model, company, year, kms_driven, fuel_type]])

            # Make predictions
            predicted_price = pipe.predict(input_data)
            st.write("Model :",model)
            st.write("Company :",company)
            st.write("Year :",year)
            st.write("Kilometres driven :",kms_driven)
            st.write("Fuel Type:",fuel_type)
            st.write("Predicted price:", predicted_price[0])

    except KeyError as e:
        st.error(f"Error: {e}. Make sure the CSV file contains the required columns.")

if __name__ == '__main__':
    main()
  
