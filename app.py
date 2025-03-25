import numpy as np
import streamlit as st
import joblib

# Load the model
model = joblib.load('model.pkl','rb')

# Streamlit UI setup
st.title("Titanic Survival Prediction App")

# Input fields
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Age = st.number_input("Age", min_value=0, max_value=100, step=1)
Sex2 = st.selectbox("Sex", ['Male', 'Female'])

# Process input data
def process_input(Pclass,Age, Sex2):
    sex_value = 1 if sex == 'Male' else 0
    return np.array([[Pclass,Age,sex_value]])

# Prediction logic
if st.button('Predict'):
    input_features = process_input(Pclass,Age,Sex2)
    prediction = model.predict(input_features)
    output = round(prediction[0], 2)

    if output == 1:
        st.success("The passenger is predicted to SURVIVE.")
    else:
        st.error("The passenger is predicted to NOT SURVIVE.")
