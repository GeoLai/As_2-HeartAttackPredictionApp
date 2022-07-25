# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:33:02 2022

@author: Lai Kar Wei
"""

import os
import pickle
import numpy as np
from PIL import Image
import streamlit as st

BEST_ESTIMATOR_SAVE_PATH = os.path.join(os.getcwd(), 'model', 'best_estimator.pkl')

with open(BEST_ESTIMATOR_SAVE_PATH, 'rb') as file:
    model = pickle.load(file)

st.sidebar.header('Heart Attack Predictor App')
select = st.sidebar.selectbox('Select Form', ['Heart Attack Questionnaire'], key='1')

st.header('What is heart attack and the causes')
st.markdown('A heart attack occurs when the flow of blood to the heart is severely reduced or blocked.') 
st.markdown('The blockage is usually due to a buildup of fat, cholesterol and other substances in the heart (coronary) arteries.')
st.markdown('The fatty, cholesterol-containing deposits are called plaques. The process of plaque buildup is called atherosclerosis.')
st.markdown('Sometimes, a plaque can rupture and form a clot that blocks blood flow. A lack of blood flow can damage or destroy part of the heart muscle.')
st.markdown('A heart attack is also called a myocardial infarction.')

st.subheader('Symptoms')
st.markdown('Common heart attack symptoms include:')
st.markdown('a. Chest pain that may feel like pressure, tightness, pain, squeezing or aching')
st.markdown('b. Pain or discomfort that spreads to the shoulder, arm, back, neck, jaw, teeth or sometimes the upper belly')
st.markdown('c. Cold sweat')
st.markdown('d. Fatigue')
st.markdown('e. Heartburn or indigestion')
st.markdown('f. Lightheadedness or sudden dizziness')
st.markdown('g. Nausea')
st.markdown('h. Shortness of breath')

st.subheader('Prediction result')

if not st.sidebar.checkbox("Hide", True, key='1'):
    st.sidebar.title('Heart Attack Prediction')
    name = st.sidebar.text_input("Name: ")
    age = st.sidebar.number_input("Age: ", 0, 100)
    thalachh =  st.sidebar.slider("Max Heart Rate Achieved: ", 50, 220)
    oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise Relative to Rest: ",float(0), float(5))
    
    submit = st.sidebar.button('Submit')
    if submit:
        new_data = np.expand_dims([age, thalachh, oldpeak],axis=0)
        outcome = model.predict(new_data)[0]

        if outcome == 0:
            st.write('Congratulation', '. You got strong heart')
            image = Image.open('heart-health.jpg')
            st.image(image)
        else:
            st.write(name, ", you got risk of heart attack! Please refer to medical professional for advice")
            st.markdown("To help you with your condition, [click](https://www.heart.org/en/healthy-living/healthy-lifestyle/prevent-heart-disease-and-stroke) here")
            image = Image.open('heart attack.jpg')
            st.image(image)

