import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Load the model and scaler
model = joblib.load('gboost_model_7.pkl')
scaler = joblib.load('scaler.pkl')

# Location mapping
location_mapping = {
            'akg nagar': 0,
            'alamcode': 1,
            'anayara': 2,
            'andoorkonam': 3,
            'anugraha': 4,
            'aradhana nagar': 5,
            'attingal': 6,
            'azhikode': 7,
            'chanthavila': 8,
            'chempazhanthy': 9,
            'chenkottukonam': 10,
            'chirayinkeezhu': 11,
            'chittattumukku': 12,
            'dhanuvachapuram': 13,
            'gandhipuram': 14,
            'gayatri': 15,
            'kaimanam': 16,
            'kakkamoola': 17,
            'kallambalam': 18,
            'kallayam': 19,
            'kalliyoor': 20,
            'kaniyapuram': 21,
            'kanjiramkulam': 22,
            'karakulam': 23,
            'karamana': 24,
            'kariavattom': 25,
            'karikkakam': 26,
            'karimanal': 27,
            'karyavattom': 28,
            'kattaikonam': 29,
            'kattakada': 30,
            'kattakkada': 31,
            'kazhakootam': 32,
            'kesavadasapuram': 33,
            'kilimanoor': 34,
            'killy': 35,
            'kollamkavu': 36,
            'kovalam': 37,
            'kowdiar': 38,
            'kulathoor': 39,
            'kunnapuzha': 40,
            'kuravankonam': 41,
            'malayinkeezhu': 42,
            'mangalapuram': 43,
            'manikanteswaram': 44,
            'mannanthala': 45,
            'maruthankuzhi': 46,
            'maruthoor': 47,
            'menamkulam': 48,
            'moongod': 49,
            'mudavanmugal': 50,
            'mukkola': 51,
            'muttathara': 52,
            'nalanchira': 53,
            'nemom': 54,
            'nettayam': 55,
            'njandoorkonam': 56,
            'ooruttambalam': 57,
            'pachalloor': 58,
            'palkulangara': 59,
            'pallichal': 60,
            'pappanamcode': 61,
            'parottukonam': 62,
            'paruthippara': 63,
            'pattom': 64,
            'peringammala': 65,
            'peroorkada': 66,
            'pettah': 67,
            'peyad': 68,
            'pidaram': 69,
            'poojappura': 70,
            'pothencode': 71,
            'pottayil': 72,
            'powdikonam': 73,
            'pravachambalam': 74,
            'premier sarayu': 75,
            'puliyarakonam': 76,
            'punnakkamughal': 77,
            'puthenthope': 78,
            'shangumukham': 79,
            'sreekariyam': 80,
            'surabhi gardens': 81,
            'thachottukavu': 82,
            'thattathumala': 83,
            'thirumala': 84,
            'udiyankulangara': 85,
            'uliyazhathura': 86,
            'ulloor': 87,
            'vattapara': 88,
            'vattiyoorkavu': 89,
            'vazhayila': 90,
            'vazhuthacaud': 91,
            'vellayani': 92,
            'venjaramoodu': 93,
            'vettamukku': 94,
            'vizhinjam': 95
        }


st.title("Plot Price Predictor")

# Input fields
selected_location = st.selectbox("Select Location", list(location_mapping.keys()))
location = location_mapping[selected_location]
property_age = st.number_input("Enter Property Age (years)", min_value=0.0)
bedrooms = st.number_input("Enter Number of Bedrooms", min_value=0)
build_area = st.number_input("Enter Build Area (sqft)", min_value=0.0)
plot_area = st.number_input("Enter Plot Area (sqft)", min_value=0.0)

# Convert Build Area and Plot Area to cents
build_area_cents = build_area / 435.6
plot_area_cents = plot_area / 435.6

if st.button("Predict Price"):
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'Plot__Beds': [bedrooms],
            'Property__Age': [property_age],
            'Build__Area': [build_area_cents],
            'Plot__Area': [plot_area_cents]
        })

        # Scale the numerical columns
        numerical_columns = ['Plot__Beds', 'Property__Age', 'Build__Area', 'Plot__Area']
        input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

        # Add location
        input_data['Plot__Location'] = [location]

        # Predict the price
        predicted_price = model.predict(input_data)[0]
        st.success(f"The Predicted Plot Price is ₹{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
