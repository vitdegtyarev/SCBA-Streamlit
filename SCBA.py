import streamlit as st
import pandas as pd
from pickle import load
import joblib
import pickle
import numpy as np
import math
from PIL import Image
import os
from config.definitions import ROOT_DIR

#ML models for predicting elastic buckling load

#Desision tree (DT)
DT_Elastic_model = joblib.load(os.path.join(ROOT_DIR, 'Cellular_Beams_Elastic_DT_2021-03-19.joblib'))
DT_Elastic_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Cellular_Beams_Elastic_DT_2021-03-19.pkl'),'rb'))

#Random forest (RF)
RF_Elastic_model = joblib.load(os.path.join(ROOT_DIR,'Cellular_Beams_Elastic_RF_2021_03_19.joblib'))
RF_Elastic_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Cellular_Beams_Elastic_RF_2021_03_19.pkl'),'rb'))

#K-nearest neighbor (KNN)
KNN_Elastic_model = joblib.load(os.path.join(ROOT_DIR,'Cellular_Beams_Elastic_KNN_2021-03-19.joblib'))
KNN_Elastic_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Cellular_Beams_Elastic_KNN_2021-03-19.pkl'),'rb'))

#Gradient boosting regressor (GBR)
GBR_Elastic_model = joblib.load(os.path.join(ROOT_DIR,'Cellular_Beams_Elastic_GBR_2021_03_20.joblib'))
GBR_Elastic_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Cellular_Beams_Elastic_GBR_2021_03_20.pkl'),'rb'))

#Extreme gradient boosting (XGBoost)
XGBoost_Elastic_model = joblib.load(os.path.join(ROOT_DIR,'Cellular_Beams_Elastic_XGBoost_2021_03_20.joblib'))
XGBoost_Elastic_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Cellular_Beams_Elastic_XGBoost_2021_03_20.pkl'),'rb'))

#Light gradient boosting machine (LightGBM)
LightGBM_Elastic_model = joblib.load(os.path.join(ROOT_DIR,'Cellular_Beams_Elastic_LightGBM_2021_03_09.joblib'))
LightGBM_Elastic_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Cellular_Beams_Elastic_LightGBM_2021_03_09.pkl'),'rb'))

#Gradient boosting with categorical features support (CatBoost)
CatBoost_Elastic_model = joblib.load(os.path.join(ROOT_DIR,'Cellular_Beams_Elastic_CatBoost_2021_03_20.joblib'))
CatBoost_Elastic_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Cellular_Beams_Elastic_CatBoost_2021_03_20.pkl'),'rb'))


#ML models for predicting ultimate load

#Desision tree (DT)
DT_Inelastic_model = joblib.load(os.path.join(ROOT_DIR,'Cellular_Beams_Inelastic_DT_2021-03-19.joblib'))
DT_Inelastic_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Cellular_Beams_Inelastic_DT_2021-03-19.pkl'),'rb'))

#Random forest (RF)
RF_Inelastic_model = joblib.load(os.path.join(ROOT_DIR,'Cellular_Beams_Inelastic_RF_2021_03_19.joblib'))
RF_Inelastic_scaler=pickle.load(open('Cellular_Beams_Inelastic_RF_2021_03_19.pkl','rb'))

#K-nearest neighbor (KNN)
KNN_Inelastic_model = joblib.load('Cellular_Beams_Inelastic_KNN_2021-03-19.joblib')
KNN_Inelastic_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Cellular_Beams_Inelastic_KNN_2021-03-19.pkl'),'rb'))

#Gradient boosting regressor (GBR)
GBR_Inelastic_model = joblib.load(os.path.join(ROOT_DIR,'Cellular_Beams_Inelastic_GBR_2021_03_13.joblib'))
GBR_Inelastic_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Cellular_Beams_Inelastic_GBR_2021_03_13.pkl'),'rb'))

#Extreme gradient boosting (XGBoost)
XGBoost_Inelastic_model = joblib.load(os.path.join(ROOT_DIR,'Cellular_Beams_Inelastic_XGBoost_2021_03_14.joblib'))
XGBoost_Inelastic_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Cellular_Beams_Inelastic_XGBoost_2021_03_14.pkl'),'rb'))

#Light gradient boosting machine (LightGBM)
LightGBM_Inelastic_model = joblib.load(os.path.join(ROOT_DIR,'Cellular_Beams_Inelastic_LightGBM_2021_03_09.joblib'))
LightGBM_Inelastic_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Cellular_Beams_Inelastic_LightGBM_2021_03_09.pkl'),'rb'))

#Gradient boosting with categorical features support (CatBoost)
CatBoost_Inelastic_model = joblib.load(os.path.join(ROOT_DIR,'Cellular_Beams_Inelastic_CatBoost_2021_03_14.joblib'))
CatBoost_Inelastic_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Cellular_Beams_Inelastic_CatBoost_2021_03_14.pkl'),'rb'))


st.header('Elastic Buckling and Ultimate Loads of Steel Cellular Beams Predicted by ML Methods')

st.sidebar.header('User Input Parameters')

image = Image.open(os.path.join(ROOT_DIR,'Cell_Beam_App.png'))
st.subheader('Dimensional Parameters')
st.image(image)


def user_input_features():
    span_length = st.sidebar.slider('L (mm)', min_value=4000, max_value=7000, step=250)
    beam_height = st.sidebar.slider('H (mm)', min_value=420, max_value=700, step=20)
    flange_width = st.sidebar.slider('Bf (mm)', min_value=162, max_value=270, step=27)
    flange_thickness = st.sidebar.slider('Tf (mm)', min_value=15, max_value=25, step=5)
    web_thickness = st.sidebar.slider('Tw (mm)', min_value=9, max_value=15, step=3) 
    height_to_diameter = st.sidebar.slider('H/Do', min_value=1.25, max_value=1.70, step=0.05)
    spacing_to_diameter = st.sidebar.slider('So/Do', min_value=1.10, max_value=1.49, step=0.03)
    yield_strength_sel = st.sidebar.radio('Fy (MPa)', ('235','355','440')) 
    if yield_strength_sel=='235': yield_strength=235
    elif yield_strength_sel=='355': yield_strength=355
    elif yield_strength_sel=='440': yield_strength=440
    
    data = {'L (mm)': span_length,
            'H (mm)': beam_height,
            'Bf (mm)': flange_width,
            'Tf (mm)': flange_thickness,
            'Tw (mm)': web_thickness,           
            'H/Do': height_to_diameter,           
            'So/Do': spacing_to_diameter,
            'Fy (MPa)': yield_strength}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()


L=df['L (mm)'].values.item()
H=df['H (mm)'].values.item()
FW=df['Bf (mm)'].values.item()
TF=df['Tf (mm)'].values.item()
TW=df['Tw (mm)'].values.item()
H_Do=df['H/Do'].values.item()
So_Do=df['So/Do'].values.item()
Do=H/H_Do
So=Do*So_Do
WP=Do*(So_Do-1)
N_holes=math.floor((L-2*Do)/So)
Led=0.5*(L-N_holes*So-Do)
Fy=df['Fy (MPa)'].values.item()


user_input={'L': "{:.0f}".format(L),
            'H': "{:.0f}".format(H),
            'Bf': "{:.0f}".format(FW),
            'Tf': "{:.0f}".format(TF),
            'Tw': "{:.0f}".format(TW),
            'Do': "{:.1f}".format(Do),
            'So': "{:.1f}".format(So),
            'WP': "{:.1f}".format(WP),
            'Led': "{:.1f}".format(Led),
            'Fy': "{:.0f}".format(Fy),
            'n': "{:.0f}".format(N_holes),
            'H/Do': "{:.2f}".format(H_Do),
            'So/Do': "{:.2f}".format(So_Do)}
user_input_df=pd.DataFrame(user_input, index=[0])
st.subheader('User Input Parameters')
st.write(user_input_df)


X_IP_Elastic=np.array([[L,H,Do,WP,FW,Led,TF,TW]])
X_IP_Inelastic=np.array([[L,H,Do,WP,FW,Fy,Led,TF,TW]])

X_IP_Elastic_DT=DT_Elastic_scaler.transform(X_IP_Elastic)
X_IP_Elastic_RF=RF_Elastic_scaler.transform(X_IP_Elastic)
X_IP_Elastic_KNN=KNN_Elastic_scaler.transform(X_IP_Elastic)
X_IP_Elastic_GBR=GBR_Elastic_scaler.transform(X_IP_Elastic)
X_IP_Elastic_XGBoost=XGBoost_Elastic_scaler.transform(X_IP_Elastic)
X_IP_Elastic_LightGBM=LightGBM_Elastic_scaler.transform(X_IP_Elastic)
X_IP_Elastic_CatBoost=CatBoost_Elastic_scaler.transform(X_IP_Elastic)

X_IP_Inelastic_DT=DT_Inelastic_scaler.transform(X_IP_Inelastic)
X_IP_Inelastic_RF=RF_Inelastic_scaler.transform(X_IP_Inelastic)
X_IP_Inelastic_KNN=KNN_Inelastic_scaler.transform(X_IP_Inelastic)
X_IP_Inelastic_GBR=GBR_Inelastic_scaler.transform(X_IP_Inelastic)
X_IP_Inelastic_XGBoost=XGBoost_Inelastic_scaler.transform(X_IP_Inelastic)
X_IP_Inelastic_LightGBM=LightGBM_Inelastic_scaler.transform(X_IP_Inelastic)
X_IP_Inelastic_CatBoost=CatBoost_Inelastic_scaler.transform(X_IP_Inelastic)

w_cr_DT=DT_Elastic_model.predict(X_IP_Elastic_DT).item()
w_cr_RF=RF_Elastic_model.predict(X_IP_Elastic_RF).item()
w_cr_KNN=KNN_Elastic_model.predict(X_IP_Elastic_KNN).item()
w_cr_GBR=GBR_Elastic_model.predict(X_IP_Elastic_GBR).item()
w_cr_XGBoost=XGBoost_Elastic_model.predict(X_IP_Elastic_XGBoost).item()
w_cr_LightGBM=LightGBM_Elastic_model.predict(X_IP_Elastic_LightGBM).item()
w_cr_CatBoost=CatBoost_Elastic_model.predict(X_IP_Elastic_CatBoost).item()

w_max_DT=DT_Inelastic_model.predict(X_IP_Inelastic_DT).item()
w_max_RF=RF_Inelastic_model.predict(X_IP_Inelastic_RF).item()
w_max_KNN=KNN_Inelastic_model.predict(X_IP_Inelastic_KNN).item()
w_max_GBR=GBR_Inelastic_model.predict(X_IP_Inelastic_GBR).item()
w_max_XGBoost=XGBoost_Inelastic_model.predict(X_IP_Inelastic_XGBoost).item()
w_max_LightGBM=LightGBM_Inelastic_model.predict(X_IP_Inelastic_LightGBM).item()
w_max_CatBoost=CatBoost_Inelastic_model.predict(X_IP_Inelastic_CatBoost).item()


st.subheader('Predicted Elastic Buckling Loads (kN/m)')
w_cr_results={'DT': "{:.1f}".format(w_cr_DT),
              'RF': "{:.1f}".format(w_cr_RF),
              'KNN': "{:.1f}".format(w_cr_KNN),
              'GBR': "{:.1f}".format(w_cr_GBR),
              'XGBoost': "{:.1f}".format(w_cr_XGBoost),
              'LightGBM': "{:.1f}".format(w_cr_LightGBM),
              'CatBoost': "{:.1f}".format(w_cr_CatBoost)}
w_cr_results_df=pd.DataFrame(w_cr_results, index=[0])


def color_col_wcr (col):
    color = '#ffff00'
    return ['background-color: %s' % color 
                if col.name=='CatBoost' 
                else ''
             for i,x in col.iteritems()]

st.dataframe(w_cr_results_df.style.apply(color_col_wcr))

st.write('CatBoost predictions showed the best agreement with the FEA results. The elastic buckling loads predicted by CatBoost are recommended for use.')


st.subheader('Predicted Ultimate Loads (kN/m)')
w_max_results={'DT': "{:.1f}".format(w_max_DT),
              'RF': "{:.1f}".format(w_max_RF),
              'KNN': "{:.1f}".format(w_max_KNN),
              'GBR': "{:.1f}".format(w_max_GBR),
              'XGBoost': "{:.1f}".format(w_max_XGBoost),
              'LightGBM': "{:.1f}".format(w_max_LightGBM),
              'CatBoost': "{:.1f}".format(w_max_CatBoost)}
w_max_results_df=pd.DataFrame(w_max_results, index=[0])



def color_col_wmax (col):
    color = '#ffff00'
    return ['background-color: %s' % color 
                if col.name=='RF' 
                else ''
             for i,x in col.iteritems()]

st.dataframe(w_max_results_df.style.apply(color_col_wmax))

st.write('RF predictions showed the best agreement with the FEA results. The ultimate loads predicted by RF are recommended for use. An appropriate safety factor should be applied.')


st.subheader('Nomenclature')
st.write('Fy is beam yield strength (MPa); n is number of evenly spaced openings along beam span; DT is decision tree; RF is random forest; KNN is k-nearest neighbor; GBR is gradient boosting regressor; XGBoost is extreme gradient boosting; LightGBM is light gradient boosting machine; CatBoost is gradient boosting with categorical features support.')

st.subheader('Reference')
st.write('Degtyarev, V.V., Tsavdaridis, K.D. Buckling and ultimate load prediction models for steel perforated beams using machine learning algorithms. Preprint')

