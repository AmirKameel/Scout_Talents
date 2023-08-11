import streamlit as st
import pandas as pd
import numpy as np
import joblib

new_df = pd.read_csv('talents.csv')
#df_data = pd.DataFrame()
# add the code to create the df_data dataframe from the data dataframe



#new_df=df_data[df_data.Age < 24]
new_df['G/90'].replace(0,1)
#y.replace(0,4)
new_df['Assist'].replace(0,1)
new_df['Pass'].replace(0,1)
new_df['PassCompleted'].replace(0,1)
new_df['PassComp%'].replace(0,5)
new_df['Tackle_Won'].replace(0,1)
new_df['AerWon'].replace(0,1)
#new_df['AerLost'].replace(0,1)
new_df['Cross'].replace(0,1)
new_df['CrossCompleted'].replace(0,1)
new_df['CrossComp%'].replace(0,2)
new_df['Talent'] = new_df['Cluster']
new_df=new_df.drop('Cluster' , axis=1)


# Add the cluster labels to the data frame
#new_df["Cluster"] = labels


st.title('Football Talents Finder')
st.write('This app helps you find the best football talents based on their nation and stats.')

nations = new_df['Nation'].unique()
nation = st.selectbox('Select a nation:', nations)

filtered_data = new_df[new_df['Nation'] == nation]
st.dataframe(filtered_data)


if st.button('predict Page'):
    # Second page
    st.title('predict if you are talent or not')

model = joblib.load('model.pkl')

# Inputs
age = st.number_input('Age') 
#age = st.number_input("Age")
mp = st.number_input("MP")
g_90 = st.number_input("G/90")
passes = st.number_input("Pass")
passcomp = st.number_input("PassCompleted")
passacc = st.number_input("PassComp%")
assist = st.number_input("Assist")
cross = st.number_input("Cross")
crosscomp = st.number_input("CrossCompleted")
crossacc = st.number_input("CrossComp%")
tacklewon = st.number_input("Tackle_Won")
aerwon = st.number_input("AerWon")

# Predict button  
if st.button('Predict'):

   prediction = model.predict([[age,mp,g_90,passes,passcomp,passacc,assist,cross,crosscomp,crossacc,tacklewon,aerwon]])
   st.write('[1] for talent , [0] for non-talent')
   st.write(prediction)


   
# Feature importance
#st.write(model.coef_)


    # Load model
