import streamlit as st 
import numpy as np 
import pandas as pd

#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.title('Machine Learning - LAPTOP PRICE PREDICTION')

st.sidebar.write("""
This is a web app demo using python libraries such as Streamlit, Sklearn etc
""")

st.sidebar.write ("For more info, please contact:")
st.sidebar.write("<a href='https://www.linkedin.com/in/mohd-sanusi-amat-sernor-9bb8b7195/'>Mohd Sanusi </a>", unsafe_allow_html=True)


data = pd.read_table('Laptop.csv', index_col = False,  sep = ',', skipinitialspace = True)
data = data.dropna()
data = data.drop(['Series','Unnamed: 0','Model'], axis = 1)

labelencoder1 = LabelEncoder()
#labelencoder2 = LabelEncoder()
labelencoder3 = LabelEncoder()
labelencoder4 = LabelEncoder()
labelencoder5 = LabelEncoder()
labelencoder6 = LabelEncoder()

data['Brand'] = labelencoder1.fit_transform(data['Brand'])
#data['Model'] = labelencoder2.fit_transform(data['Model'])
data['Processor'] = labelencoder3.fit_transform(data['Processor'])
data['Processor_Gen'] = labelencoder4.fit_transform(data['Processor_Gen'])
data['Hard_Disk_Capacity'] = labelencoder5.fit_transform(data['Hard_Disk_Capacity'])
data['OS'] = labelencoder6.fit_transform(data['OS'])

X = data.drop('Price', axis = 1)
y = data['Price']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)


RandomForest = RandomForestClassifier()
RandomForest.fit(Xtrain, ytrain)
ypred = RandomForest.predict(Xtest)

st.write(ypred)

D1 = st.sidebar.slider('Brand', 0.01, 10.0,value=1.0)
D2 = st.sidebar.slider('Processor', 0.01, 10.0,value=1.0)
D3 = st.sidebar.slider('Processor_Gen', 0.01, 10.0,value=1.0)
D4 = st.sidebar.slider('Hard_Disk_Capacity', 0.01, 10.0,value=1.0)
D5 = st.sidebar.slider('OS', 0.01, 10.0,value=1.0)            

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
