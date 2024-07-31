import streamlit as st
import joblib
import numpy as np

# Load the model
loaded_model = joblib.load('dtf_model.joblib')

st.title('Social Network Ads Prediction')

st.write('What do you want to find out?')
user_input = st.text_input('Please enter your query or question')

if st.checkbox('Click Next', key='next_checkbox'):
    st.write('Please enter the following')

    def get_user_input():
        Age = st.number_input('Age', value=None, placeholder="Type your age")
        Estimated_Salary = st.number_input('Estimated Salary', value=None, placeholder="Type your estimated salary")
        new_gender = st.number_input('Gender', value=None, placeholder='Enter your gender (0 for Male, 1 for Female)')
        return Age, Estimated_Salary, new_gender

    Age, Estimated_Salary, new_gender = get_user_input()

if st.button('Prediction', key='predict_button'):
    prediction = loaded_model.predict(np.array([[Age, Estimated_Salary, new_gender]]))

    if prediction == 0:
        st.write('User will not purchase')
    else:
        st.write('User will make a purchase')

    


