import streamlit as st 
import numpy as np 
import pandas as pd
import pickle

#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

st.title('Machine Learning - SMARTPHONE PRICE CLASSIFICATION')
st.write("""
This is a web app demo used to help classifying smartphone prices based on their specifications between Low-Range, Mid-Range, High-End and Top-End.
""")
st.sidebar.write("""
This is a web app demo using python libraries such as Streamlit, Sklearn etc
""")

st.sidebar.write ("For more info, please contact:")
st.sidebar.write("<a href='https://www.linkedin.com/in/mohd-sanusi-amat-sernor-9bb8b7195/'>Mohd Sanusi </a>", unsafe_allow_html=True)


data = pd.read_table('phone.csv', index_col = False,  sep = ',', skipinitialspace = True)
data = data.dropna()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = data.drop('price_range', axis = 1)
y = data['price_range']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(Xtrain, ytrain)
ypred = knn.predict(Xtest)

pickle_out = open('knn.pkl','wb')
pickle.dump(knn, pickle_out)
pickle_out.close()

pickle_in = open('knn.pkl','rb')
classifier = pickle.load(pickle_in)

st.header('Phone Specifications')
battery_power = st.slider('battery_power', 500, 4000,value=1000)
clock_speed = st.slider('clock_speed', 0.3, 3.5,value=1.0)
n_cores = st.slider('n_cores', 1, 8,value=1)
ram = st.slider('ram', 400, 6000,value=1000)
int_memory = st.slider('int_memory', 4, 128,value=8)
blue = st.slider('blue', 0, 1,value=1)
dual_sim = st.slider('dual_sim', 0, 1,value=0)
fc = st.slider('fc', 0, 20,value=10)
four_g = st.slider('four_g', 0, 1,value=1)
m_dep = st.slider('m_dep', 0, 1.5,value=1)
mobile_wt = st.slider('mobile_wt', 50, 250,value=100)
pc = st.slider('pc', 1, 20,value=5)
px_height = st.slider('px_height', 20, 2000,value=900)
px_width = st.slider('px_width', 20, 2500,value=1200)
sc_h = st.slider('sc_h', 5, 20,value=10)
sc_w = st.slider('sc_w', 1, 20,value=7)
talk_time = st.slider('talk_time', 1, 24,value=10)
three_g = st.slider('three_g', 0, 1,value=1)
touch_screen = st.slider('touch_screen', 0, 1,value=1)
wifi = st.slider('wifi', 0, 1,value=1)
price_range = st.slider('price_range', 0, 3,value=1)

prediction = classifier.predict([battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep,	mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi, price_range])
if prediction == 0:
  st.write('This is a Low-Range Phone')
elif prediction == 1:
  st.write('This is a Mid-Range Phone')
elif prediction == 2:
  st.write('This is a High-End Phone')
else:
  st.write('This is a Flagship Phone')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
