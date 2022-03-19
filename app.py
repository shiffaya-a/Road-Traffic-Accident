import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import encode, getPredict_Model



model = joblib.load(r'Model/RandomForestModel.joblib')

st.set_page_config(page_title="Accident Severity Prediction App",page_icon="🚧", layout="centered")
#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']

options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']
options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']

options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
       'other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)', 'Unknown']
options_type_Of_junction =['Y Shape','No junction','Crossing','Other','Unknown','Shape','T Shape','X Shape']
options_road_surface_type=['Asphalt roads','Earth roads','Gravel roads','Other', 'Asphalt roads with some distress']
options_type_of_collision=['vehicle_with_vehicle_collision','Collision with roadside objects','Collision with pedestrians',
'Rollover','Collision with animals','Collision with roadside-parked vehicles','Fall from vehicles',
'Other','Unknown','With Train']
options_light_conditions=['Daylight','Darkness','Darkness - no lighting','Darkness - lights unlit']
options_vehicle_movement=['Going straight','Moving Backward','Other','Reversing',
'Turnover','Getting off','Entering a junction','Overtaking','Unknown','Stopping','U-Turn','Waiting to go','Parked']
options_age_band_of_casualty=['na','18-30','31-50','Under 18','Over 51','5']
options_vehicle_driver_relation=['Employee''Owner''Other''Unknown']
options_casualty_class =['Driver or rider','na','Pedestrian','Passenger']
options_service_year_of_vehicle=['Unknown','2-5yrs','Above 10yr','5-10yrs','1-2yr','Below 1yr']
options_Age_band_of_driver=['18-30','31-50','Over 51','Unknown','Under 18']


features=['Number_of_casualties','Number_of_vehicles_involved','Time','Type_of_collision','Light_conditions',
'Vehicle_movement','Cause_of_accident','Age_band_of_casualty','Vehicle_driver_relation','Age_band_of_driver',
'Area_accident_occured','Road_surface_type','Service_year_of_vehicle','Casualty_class','Type_of_vehicle',
'Day_of_week','Types_of_Junction','Lanes_or_Medians']


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
        Area_accident_occured=st.selectbox("Area accident occured",options=options_acc_area)

        Road_surface_type=st.selectbox("Road surface type",options=options_road_surface_type)
        Service_year_of_vehicle=st.selectbox("Service year of vehicle",options=options_service_year_of_vehicle)
        Casualty_class=st.selectbox("Casualty class",options=options_casualty_class)
        Type_of_vehicle=st.selectbox("Type of vehicle",options=options_vehicle_type)

        Day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
        Types_of_Junction=st.selectbox("Types of Junction",options=options_type_Of_junction)
        Lanes_or_Medians=st.selectbox("Lanes or Medians",options=options_lanes)

        submit = st.form_submit_button("Predict")

        if submit:
            Type_of_collision=encode(Type_of_collision,options_type_of_collision)
            Light_conditions=encode(Light_conditions,options_light_conditions)
            Vehicle_movement=encode(Vehicle_movement,options_vehicle_movement)
            Cause_of_accident = encode(Cause_of_accident, options_cause)

            Age_band_of_casualty=encode(Age_band_of_casualty,options_age_band_of_casualty)
            Vehicle_driver_relation=encode(Vehicle_driver_relation,options_vehicle_driver_relation)
            Age_band_of_driver =  encode(Age_band_of_driver, options_age)
            Area_accident_occured =  encode(Area_accident_occured, options_acc_area)

            Road_surface_type=encode(Road_surface_type,options_road_surface_type)
            Service_year_of_vehicle=encode(Service_year_of_vehicle,options_service_year_of_vehicle)
            Casualty_class=encode(Casualty_class,options_casualty_class)
            Type_of_vehicle = encode(Type_of_vehicle, options_vehicle_type)

            Day_of_week = encode(Day_of_week, options_day)
            Types_of_Junction=encode(Types_of_Junction,options_type_Of_junction)
            Lanes_or_Medians = encode(Lanes_or_Medians, options_lanes)

            data=np.array([Number_of_casualties,Number_of_vehicles_involved,Time,Type_of_collision,Light_conditions,Vehicle_movement,Cause_of_accident,
            Age_band_of_casualty,Vehicle_driver_relation,Age_band_of_driver,Area_accident_occured,Road_surface_type,
            Service_year_of_vehicle,Casualty_class,Type_of_vehicle,Day_of_week,Types_of_Junction,Lanes_or_Medians]).reshape(1,-1)
            pred = getPredict_Model(data=data, model=model)
            if pred[0] is 0:
                result = 'Fatal Injury'
            elif pred[0] is 1:
                result = 'Serious Injury'
            else:
                result = 'Slight Injury'
            st.write(f"The predicted severity is: {result}")

if __name__ == '__main__':
    main()
