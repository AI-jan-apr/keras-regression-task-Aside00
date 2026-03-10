import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Prediction AI")
st.write("Enter the house details below to estimate the market price.")

@st.cache_resource 
def load_assets():
    model = load_model('house_model.keras')
    scaler = joblib.load('scaler_weights.pkl')
    return model, scaler

model, scaler = load_assets()

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", min_value=1, value=3)
    sqft_living = st.number_input("Living Area (sqft)", min_value=500, value=1800)
    floors = st.number_input("Floors", min_value=1, value=1)
    view = st.slider("View Score (0-4)", 0, 4, 0)
    grade = st.slider("Grade (1-13)", 1, 13, 7)
    lat = st.number_input("Latitude", value=47.5)
    sqft_living15 = st.number_input("Living Area of Neighbors", value=1800)
    year = st.number_input("Year Sold", value=2014)

with col2:
    bathrooms = st.number_input("Bathrooms", min_value=1.0, value=2.0, step=0.5)
    sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, value=5000)
    waterfront = st.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    condition = st.slider("Condition (1-5)", 1, 5, 3)
    sqft_above = st.number_input("Sqft Above Ground", value=1800)
    sqft_basement = st.number_input("Sqft Basement", value=0)
    yr_built = st.number_input("Year Built", value=1990)
    yr_renovated = st.number_input("Year Renovated (0 if none)", value=0)
    zipcode = st.number_input("Zipcode", value=98001)
    long = st.number_input("Longitude", value=-122.2)
    sqft_lot15 = st.number_input("Lot Area of Neighbors", value=5000)
    month = st.slider("Month Sold", 1, 12, 5)

if st.button("💰 Predict Price"):
    features = [
        bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, 
        condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, 
        zipcode, lat, long, sqft_living15, sqft_lot15, year, month
    ]
    
    data = np.array(features).reshape(1, -1)
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data.astype('float32'), verbose=0)
    
    st.success(f"### Estimated Price: ${prediction[0][0]:,.2f}")