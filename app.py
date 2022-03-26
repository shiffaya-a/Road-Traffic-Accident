import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import encode, getPredict_Model
from load_model import get_model
import requests

model=get_model(model_path=r'Model/RandomForestModel.joblib')

st.set_page_config(page_title="Accident Severity Prediction App",page_icon="🚧", layout="centered")
#creating option list for dropdown menu
options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']


options_type_of_collision=['vehicle_with_vehicle_collision','Collision with roadside objects','Collision with pedestrians',
'Rollover','Collision with animals','Collision with roadside-parked vehicles','Fall from vehicles',
'Other','Unknown','With Train']
options_light_conditions=['Daylight','Darkness','Darkness - no lighting','Darkness - lights unlit']
options_vehicle_movement=['Going straight','Moving Backward','Other','Reversing',
'Turnover','Getting off','Entering a junction','Overtaking','Unknown','Stopping','U-Turn','Waiting to go','Parked']
options_age_band_of_casualty=['na','18-30','31-50','Under 18','Over 51','5']
options_vehicle_driver_relation=['Employee''Owner''Other''Unknown']
options_Age_band_of_driver=['18-30','31-50','Over 51','Unknown','Under 18']


features=['Number_of_casualties','Number_of_vehicles_involved','Time','Type_of_collision','Light_conditions',
'Vehicle_movement','Cause_of_accident','Age_band_of_casualty','Vehicle_driver_relation','Age_band_of_driver',
]


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")

        Number_of_casualties=st.slider("Select No of Casualities",1,8,value=0,format="%d")
        Number_of_vehicles_involved=st.slider("Select No of Vehicles:",1,7,value=0,format="%d")
        Time = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")

        Type_of_collision=st.selectbox("Collision Type",options=options_type_of_collision)
        Light_conditions=st.selectbox("Light Condtions",options=options_light_conditions)
        Vehicle_movement=st.selectbox("Vehicle Movement",options=options_vehicle_movement)
        Cause_of_accident=st.selectbox("Cause of Accident",options=options_cause)

        Age_band_of_casualty=st.selectbox("Age band of casuality",options=options_age_band_of_casualty)
        Vehicle_driver_relation=st.selectbox("Vehicle driver relation",options=options_vehicle_driver_relation)
        Age_band_of_driver=st.selectbox("Age band of driver",options=options_Age_band_of_driver)

        submit = st.form_submit_button("Predict")

        if submit:
            Type_of_collision=encode(Type_of_collision,options_type_of_collision)
            Light_conditions=encode(Light_conditions,options_light_conditions)
            Vehicle_movement=encode(Vehicle_movement,options_vehicle_movement)
            Cause_of_accident = encode(Cause_of_accident, options_cause)

            Age_band_of_casualty=encode(Age_band_of_casualty,options_age_band_of_casualty)
            Vehicle_driver_relation=encode(Vehicle_driver_relation,options_vehicle_driver_relation)
            Age_band_of_driver =  encode(Age_band_of_driver, options_Age_band_of_driver)

            data=np.array([Number_of_casualties,Number_of_vehicles_involved,Time,Type_of_collision,Light_conditions,Vehicle_movement,Cause_of_accident,
            Age_band_of_casualty,Vehicle_driver_relation,Age_band_of_driver]).reshape(1,-1)
            pred = getPredict_Model(data=data, model=model)
            if pred[0] == 0:
                result = 'Fatal Injury'
            elif pred[0] == 1:
                result = 'Serious Injury'
            else:
                result = 'Slight Injury'
            st.write(f"The predicted severity is: {result}")

if __name__ == '__main__':
    main()
