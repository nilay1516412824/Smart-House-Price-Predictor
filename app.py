import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load best model
model = joblib.load(r'C:\nilay code\Projects\shp\best_house_price_model.pkl')

st.set_page_config(page_title="Smart House Price Predictor", page_icon="🏡", layout="centered")

st.title("🏡 Smart House Price Predictor")
st.markdown("### Predict house prices using AI models trained on housing data.")

st.write("Enter the details of the house below:")

# User Inputs
size = st.number_input("House Size (sqft)", min_value=500, max_value=5000, step=100, value=2000)
rooms = st.slider("Number of Rooms", 1, 6, 3)
location = st.slider("Location Score (1-10)", 1, 10, 5)
age = st.slider("Age of House (years)", 0, 50, 10)

if st.button("🔮 Predict Price"):
    input_data = np.array([[size, rooms, location, age]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"Estimated Price: **${prediction:,.2f}**")
    
    # Confidence Interval (for Random Forest)
    if hasattr(model, "estimators_"):
        preds = [est.predict(input_data)[0] for est in model.estimators_]
        low, high = np.percentile(preds, 5), np.percentile(preds, 95)
        st.info(f"Confidence Interval: ${low:,.2f} - ${high:,.2f}")
