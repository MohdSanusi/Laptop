import streamlit as st 
import numpy as np 
import pandas as pd

#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier
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

#RandomForest = RandomForestClassifier()
#RandomForest.fit(Xtrain, ytrain)
#ypred = RandomForest.predict(Xtest)

D1 = st.slider('battery_power', 500, 4000,value=1000)
D2 = st.slider('clock_speed', 0.3, 3.5,value=1.0)
D3 = st.slider('n_cores', 1, 8,value=1)
D4 = st.slider('ram', 400, 6000,value=1000)
D5 = st.slider('int_memory', 4, 128,value=8)

st.write(ypred)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
