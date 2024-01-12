import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model 
lr = pickle.load(open('LR.pkl','rb'))

#load dataset
data = pd.read_csv('Breast_Cancern.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Breast Cancern')

html_layout1 = """
<br>
<div style="background-color:blue ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Breast Cancern Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['Logistic Regression','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset kaggle</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDA'):
    pr =ProfileReport(data,explorative=True)
    st.header('*Input Dataframe*')
    st.write(data)
    st.write('---')
    st.header('*Profiling Report*')
    st_profile_report(pr)

#train test split
X = data.drop('diagnosis',axis=1)
y = data['diagnosis']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    radius_mean = st.sidebar.slider('radius_mean',float(data['radius_mean'].min()),float(data['radius_mean'].max()))
    texture_mean = st.sidebar.slider('texture_mean',float(data['texture_mean'].min()),float(data['texture_mean'].max()))
    perimeter_mean = st.sidebar.slider('perimeter_mean',float(data['perimeter_mean'].min()),float(data['perimeter_mean'].max()))
    area_mean = st.sidebar.slider('area_mean',float(data['area_mean'].min()),float(data['area_mean'].max()))
    smoothness_mean = st.sidebar.slider('smoothness_mean',float(data['smoothness_mean'].min()),float(data['smoothness_mean'].max()))
    compactness_mean = st.sidebar.slider('compactness_mean',float(data['compactness_mean'].min()),float(data['compactness_mean'].max()))
    concavity_mean = st.sidebar.slider('concavity_mean',float(data['concavity_mean'].min()),float(data['concavity_mean'].max()))
    concave_points_mean = st.sidebar.slider('concave_points_mean',float(data['concave_points_mean'].min()),float(data['concave_points_mean'].max()))
    symmetry_mean = st.sidebar.slider('symmetry_mean',float(data['symmetry_mean'].min()),float(data['symmetry_mean'].max()))
    fractal_dimension_mean = st.sidebar.slider('fractal_dimension_mean',float(data['fractal_dimension_mean'].min()),float(data['fractal_dimension_mean'].max()))
    radius_se = st.sidebar.slider('radius_se',float(data['radius_se'].min()),float(data['radius_se'].max()))
    texture_se = st.sidebar.slider('texture_se',float(data['texture_se'].min()),float(data['texture_se'].max()))
    perimeter_se = st.sidebar.slider('perimeter_se',float(data['perimeter_se'].min()),float(data['perimeter_se'].max()))
    area_se = st.sidebar.slider('area_se',float(data['area_se'].min()),float(data['area_se'].max()))
    smoothness_se = st.sidebar.slider('smoothness_se',float(data['smoothness_se'].min()),float(data['smoothness_se'].max()))
    compactness_se = st.sidebar.slider('compactness_se',float(data['compactness_se'].min()),float(data['compactness_se'].max()))
    concavity_se = st.sidebar.slider('concavity_se',float(data['concavity_se'].min()),float(data['concavity_se'].max()))
    concave_points_se = st.sidebar.slider('concave_points_se',float(data['concave_points_se'].min()),float(data['concave_points_se'].max()))
    symmetry_se = st.sidebar.slider('symmetry_se',float(data['symmetry_se'].min()),float(data['symmetry_se'].max()))
    fractal_dimension_se = st.sidebar.slider('fractal_dimension_se',float(data['fractal_dimension_se'].min()),float(data['fractal_dimension_se'].max()))
    radius_worst = st.sidebar.slider('radius_worst',float(data['radius_worst'].min()),float(data['radius_worst'].max()))
    texture_worst = st.sidebar.slider('texture_worst',float(data['texture_worst'].min()),float(data['texture_worst'].max()))
    perimeter_worst = st.sidebar.slider('perimeter_worst',float(data['perimeter_worst'].min()),float(data['perimeter_worst'].max()))
    area_worst = st.sidebar.slider('area_worst',float(data['area_worst'].min()),float(data['area_worst'].max()))
    smoothness_worst = st.sidebar.slider('smoothness_worst',float(data['smoothness_worst'].min()),float(data['smoothness_worst'].max()))
    compactness_worst = st.sidebar.slider('compactness_worst',float(data['compactness_worst'].min()),float(data['compactness_worst'].max()))
    concavity_worst = st.sidebar.slider('concavity_worst',float(data['concavity_worst'].min()),float(data['concavity_worst'].max()))
    concave_points_worst = st.sidebar.slider('concave_points_worst',float(data['concave_points_worst'].min()),float(data['concave_points_worst'].max()))
    symmetry_worst = st.sidebar.slider('symmetry_worst',float(data['symmetry_worst'].min()),float(data['symmetry_worst'].max()))
    fractal_dimension_worst = st.sidebar.slider('fractal_dimension_worst',float(data['fractal_dimension_worst'].min()),float(data['fractal_dimension_worst'].max()))

    
    user_report_data = {
        'radius_mean':radius_mean,
        'texture_mean':texture_mean,
        'perimeter_mean':perimeter_mean,
        'area_mean':area_mean,
        'smoothness_mean':smoothness_mean,
        'compactness_mean':compactness_mean,
        'concavity_mean':concavity_mean,
        'concave_points_mean':concave_points_mean,
        'symmetry_mean':symmetry_mean,
        'fractal_dimension_mean':fractal_dimension_mean,
        'radius_se':radius_se,
        'texture_se':texture_se,
        'perimeter_se':perimeter_se,
        'area_se':area_se,
        'smoothness_se':smoothness_se,
        'compactness_se':compactness_se,
        'concavity_se':concavity_se,
        'concave_points_se':concave_points_se,
        'symmetry_se':symmetry_se,
        'fractal_dimension_se':fractal_dimension_se,
        'radius_worst':radius_worst,
        'texture_worst':texture_worst,
        'perimeter_worst':perimeter_worst,
        'area_worst':area_worst,
        'smoothness_worst':smoothness_worst,
        'compactness_worst':compactness_worst,
        'concavity_worst':concavity_worst,
        'concave_points_worst':concave_points_worst,
        'symmetry_worst':symmetry_worst,
        'fractal_dimension_worst':fractal_dimension_worst

    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = svm.predict(user_data)
lr_score = accuracy_score(y_test,lr.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena kanker'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(lr_score*100)+'%')