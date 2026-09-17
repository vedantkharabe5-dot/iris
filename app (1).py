
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('iris_model.pkl')

# Page title
st.title("Iris Species Prediction")

st.write("Enter the measurements of the iris flower to predict its species.")

# Input labels for features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0)

# Prediction button
if st.button("Predict Species"):
    input_data = np.array([[sepal_length,
                             sepal_width,
                             petal_length,
                             petal_width]]).astype(np.float64)
    
    prediction = model.predict(input_data)
    st.success(f"The predicted Iris species is: {prediction[0]}")
