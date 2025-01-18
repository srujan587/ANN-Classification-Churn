import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st


## Load Model
model= load_model('model.h5')

## Load Encoders and Sclers
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('one_hot_encoder_geography.pkl','rb') as file:
   one_hot_encoder_geography=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

##Streamlit App

st.write('Customer Churn Prediction')

georaphy=st.selectbox('Geography',one_hot_encoder_geography.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age= st.slider('Age',18,92)
balance= st.number_input('Balance')
credit_score= st.number_input('Credit Score')
estimated_salary= st.number_input('Estimated Salary')
tenure= st.slider('Tenure',0,10)
number_of_products= st.slider('Number of Products',1,4)
has_cr_card= st.selectbox("Has Credit Card",[0,1])
is_active_member= st.selectbox("Is Active Member",[0,1])

## Convert to Input Data

input_data= pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' :[balance],
    'NumOfProducts' : [number_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' :[is_active_member],
    'EstimatedSalary' : [estimated_salary]    
})


##Transform Geography

geo_encoder = one_hot_encoder_geography.transform([[georaphy]])
geo_encoder_pd=pd.DataFrame(geo_encoder.toarray(), columns= one_hot_encoder_geography.get_feature_names_out(['Geography']))

input_data= pd.concat([input_data.reset_index(drop=True),geo_encoder_pd],axis=1)

##Scaling the data

scaled_data=scaler.transform(input_data)

## Predict te churn

prediction= model.predict(scaled_data)

## Prediction Prob

prediction_pro = prediction[0][0]

st.write('Prediction Probbility',prediction_pro)

if prediction_pro >0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')