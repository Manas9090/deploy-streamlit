import numpy as np
import streamlit as st
import joblib

# Load the model
model = joblib.load('model.pkl')

# Streamlit UI setup
st.title("Titanic Survival Prediction App")

# Input fields
age = st.number_input("Age", min_value=0, max_value=100, step=1)
SibSp = st.number_input("SibSp", min_value=0, max_value=100, step=1) 
Parch = st.number_input("Parch", min_value=0, max_value=100, step=1)
Fare = st.number_input("Fare", min_value=0, max_value=100, step=1)

sex = st.selectbox("Sex", ['Male', 'Female'])
class_ = st.selectbox("Passenger Class", [1, 2, 3])
class2_ = st.selectbox("Passenger Class", [1, 2, 3])

# Process input data
def process_input(age, SibSp, Parch, Fare, sex, class_,class2_):
    sex_value = 1 if sex == 'Male' else 0
    return np.array([[age,SibSp,Parch,Fare,sex_value, class_, class2_]])

# Prediction logic
if st.button('Predict'):
    input_features = process_input(age, SibSp, Parch, Fare, sex, class_,class2_)
    prediction = model.predict(input_features)
    output = round(prediction[0], 2)

    if output == 1:
        st.success("The passenger is predicted to SURVIVE.")
    else:
        st.error("The passenger is predicted to NOT SURVIVE.")
