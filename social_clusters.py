import os
import glob
import pandas as pd
import numpy as np
import pickle
import joblib
# K-mean clustering libraries
from kmodes.kprototypes import KPrototypes
# import minmax scaler
from sklearn.preprocessing import MinMaxScaler
from src.utils.functions import validation

model_dir = 'models/'
scaler_dir = 'scalers/'
input_dir = 'data/'

types = ['Zwembad', 'Administratief centrum', 'Cultureel centrum','Museum', 'RVT/WZC/revalidatiecentrum','Technische middelbare school', 'Bibliotheek', 'Sporthal','Academie', 'Stadhuis/Gemeentehuis', 'Ontmoetingscentrum','Andere gebouwen', 'Sportcomplex', 'Algemene middelbare school','Ziekenhuis', 'Lagere school', 'Brandweerkazerne', 'Stadion','Werkplaats', 'OCMW Woningen','Buitengewoon lager onderwijs (MPI)', 'Politiegebouw', 'Jeugdhuis','Dienstencentrum/CAW/dagverblijf','Buitengewoon middelbaar onderwijs (BUSO)', 'Kleuterschool','OCMW Administratief centrum', 'Kast', 'Kinderdagverblijf/BKO/IBO','Laadeiland', 'Voetbalveld', 'Kerk', 'Pomp', 'Andere terreinen','Parking', 'Fontein', 'Tennisveld', 'Containerpark', 'Andere','School', 'Straatverlichting', 'Looppiste', 'Park']

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# class Profile():
#     """
#     A class to create synthetic load profiles
#     """
#     def __init__(
#         self,
#         params: dict,
#     ):

#     self.params = params

@st.cache
def load_data():
    st_p = pd.read_csv(input_dir+'st_p_kproto10.csv', index_col=0, parse_dates=[0])
    # drop nan and inf values
    st_p.dropna(inplace=True)
    # drop inf values
    st_p.drop(st_p[st_p.values == np.inf].index, inplace=True)
    st_p.drop(st_p[st_p.index.duplicated()].index, inplace=True)
    return st_p

def load_model():
    model = pickle.load(open(model_dir+'kproto10.pkl',  "rb"))
    return model

#@st.cache
def load_metadata():
    # read a txt file with the metadata
    meta = open(input_dir+'types.txt', 'r').read().split(',')
    return meta

if __name__ == "__main__":
    st.title("Profile Clustering")
    profiles = load_data()
    kproto = load_model()
    scaler = joblib.load(scaler_dir+'scaler.gz')
    # double-ended slider morning/evening
    evening = st.slider('Use of building in the evening:', 0.0, 1.0, 0.8, 0.01)
    if 0 <= evening < 0.25:
        st.markdown("The building is **_barely_ used** in the evening")
    elif 0.25 <= evening < 0.5:
        st.markdown("The building is **_sometimes_ used** in the evening")
    elif 0.5 <= evening < 0.75:
        st.markdown("The building is **_often_ used** in the evening")
    elif 0.75 <= evening <= 1.0:
        st.markdown("The building is **_mostly_ used** in the evening")
    # double-ended slider morning/evening
    weekend = st.slider('Use of building in the weekends:', 0.0, 1.0, 0.2, 0.01)
    if 0 <= weekend < 0.25:
        st.markdown("The building is **_barely_ used** in the weekends")
    elif 0.25 <= weekend < 0.5:
        st.markdown("The building is **_sometimes_ used** in the weekends")
    elif 0.5 <= weekend < 0.75:
        st.markdown("The building is **_often_ used** in the weekends")
    elif 0.75 <= weekend <= 1.0:
        st.markdown("The building is **_mostly_ used** in the weekends")
    # Enter yearly consumption in float
    yearly_consumption = st.number_input('Yearly consumption [kWh]:', min_value=0, max_value=500000, value=130000, step=100)
    # scale yearly consumption using minmax scaler
    scaled_consumption = scaler.transform([[yearly_consumption]])[0][0]
    #st.write(scaled_consumption)
    # Dropdown list for the type of building
    building_type = st.selectbox('Type of building:', types,  index=1)
    #st.write(kproto)
    row = np.array([scaled_consumption, weekend, evening, building_type])
    #st.write(np.shape(row.reshape(1,-1)))
    cluster = kproto.predict(row.reshape(1,-1), categorical=[3])
    #st.write(row)
    # markdown title
    st.markdown("## Predicted cluster: " + str(cluster[0]))
    #st.write(profiles.columns)
    ts = profiles[str(cluster[0])] * yearly_consumption
    day_p = ts.groupby(ts.index.hour).mean()
    # set ggplot style in matplotlib
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    day_p.plot(ax = ax)
    ax.set_xlabel('Hour')
    # grid on
    ax.grid(True)
    plt.title('Average Day Profile', fontsize=18, loc='left')
    # y axis label in kWh
    ax.set_ylabel('kWh', fontsize=15)
    st.pyplot(fig)
    week_dist = ts.groupby(ts.index.weekday).sum() 
    # divide by number of weeks in ts
    week_dist = week_dist / len(ts.resample('W').count())
    #st.write(week_dist)
    #st.write(len(ts.resample('W').count()))
    fig, ax = plt.subplots()
    # set size
    fig.set_size_inches(10, 5)
    # bar chart for the weekdays
    week_dist.plot(ax = ax, kind='bar')
    # set x tick labels as weekday names
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.title('Weekday distribution', fontsize=18, loc='left')
    ax.set_ylabel('kWh', fontsize=18)
    ax.set_xlabel('Weekday')
    # x axis ticks to show the string names of the weekdays
    st.pyplot(fig)
    st.write('Download the profile as a CSV file')
    st.download_button('Download profile', ts,  filename='profile_cluster{}.csv'.format(cluster[0]))
    