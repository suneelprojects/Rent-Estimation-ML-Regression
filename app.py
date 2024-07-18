############## Libraries/Modules ##############

# Basic Libraries
import pandas as pd
import numpy as np

# ModelLoading Libraries
import joblib

# UI & Logic Library
import streamlit as st

####################### Loading Trained Model Files #########
ohe = joblib.load("ohe.pkl") # converting categorical x to numerical
sc = joblib.load("sc.pkl") # converting numeric cols under one scale
poly = joblib.load("poly.pkl") # convert x features to poly features
model = joblib.load("rent_polyreg.pkl") # trained poly regression file

########################## UI Code ################################

st.header("Rent Estimation for the Given Features.")

# Dividing Row into columns in streamlit window
p1, p2, p3 = st.columns(3)

with p2:
    st.image("pic.jpg")

st.write("This app built on the below features to estimate rent value.")

df = pd.read_csv("RentInputData.csv")

st.dataframe(df.head(5))

st.subheader("Enter Property Details to Estimate Rent:")

# General Input
# sqft = st.number_input("Enter Sqft Value:")

# Form Type Input
col1, col2, col3 = st.columns(3)
with col1:
    city = st.selectbox("City:", df.City.unique())
with col2:
    area = st.selectbox("Area:", df.AreaType.unique())
with col3:
    sqft = st.number_input("Sqft:")

col4, col5, col6, col7 = st.columns(4)
with col4:
    bed = st.number_input("Bedrooms:")
with col5:
    bath = st.number_input("Bathrooms:")
with col6:
    floor = st.number_input("Floor No:")
with col7:
    nooffloors = st.number_input("Total No Of Floors:")

col8, col9, col10 = st.columns(3)
with col8:
    furnish = st.selectbox("Furnishing Status:", df.FurnishingStatus.unique())
with col9:
    tenant = st.selectbox("Tenant Preferred:", df.TenantPreferred.unique())
with col10:
    poc = st.selectbox("PointOfContact:", df.PointofContact.unique())

###################### Logic Code #############################

if st.button("EstimateRent"):

    row = pd.DataFrame([[city, area, sqft, bed, bath, floor, nooffloors, furnish, tenant, poc]], columns=df.columns)
    st.write("Given Input Data:")
    st.dataframe(row)
    
    # Applying Feature Modification steps before giving it to model
    
    row.FurnishingStatus.replace({'semi-furnished':46.65, 'unfurnished':38.28,'furnished':15.07}, inplace=True)
    row.TenantPreferred.replace({'bachelors/family':71.21,'bachelors':18.31,'family':10.48}, inplace=True)
    row.PointofContact.replace({'owner':68,'agent':32,'builder':0}, inplace=True)
    
    # Onehot Encoding
    row_ohe = ohe.transform(row[['City','AreaType']]).toarray()
    row_ohe = pd.DataFrame(row_ohe, columns=ohe.get_feature_names_out())
    
    row = pd.concat([row.iloc[:, 2:], row_ohe], axis=1)
    
    # Scaling
    row.iloc[:, 0:8] = sc.transform(row.iloc[:, 0:8])
    
    # Converting to polynomial features
    
    row_poly = poly.transform(row)
    
    rent = round(model.predict(row_poly)[0],2)
    
    st.write(f"Estimated Rent Value: ₹ {rent} k")

    

