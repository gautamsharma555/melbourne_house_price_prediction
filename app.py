import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import math
import os  # Import the os module

# Load the data
@st.cache_data
def load_data():
    # Check if the file exists in the current directory
    data_file = 'Melbourne_housing_FULL.csv'
    if not os.path.exists(data_file):
        # If the file doesn't exist, try the path relative to the script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(script_dir, 'Melbourne_housing_FULL.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Could not find the data file at either '{os.path.abspath('Melbourne_housing_FULL.csv')}' or '{data_file}'.  Please make sure the data file is in the same directory as the script or provide the correct path.")
    data = pd.read_csv(data_file)
    df = data.copy()
    return df

# Preprocess the data
@st.cache_data
def preprocess_data(df):
    # Drop Address (high cardinality), Date (not needed for prediction)
    df = df.drop(['Address', 'Date'], axis=1)

    # Handle missing values - using median for numerical, mode for categorical
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

   # Convert categorical columns to numerical using LabelEncoder
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# Train the model
@st.cache_resource
def train_model(df):
    # Separate features and target
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Gradient Boosting Regressor model.  Use a simplified set of parameters for faster training.
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42) # Reduced n_estimators
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, r2

def main():
    st.title('Melbourne Housing Price Prediction')
    st.write("Predicting housing prices in Melbourne using Gradient Boosting Regressor.")

    try:
        df = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        return  # Stop if the data file is not found

    df = preprocess_data(df)  # Preprocess the data

    # Display some info about the data
    st.write("First few rows of the preprocessed data:")
    st.dataframe(df.head())

    st.write("Data summary:")
    st.write(df.describe())

    # Train the model
    model, X_test, y_test = train_model(df)

    # Evaluate the model
    mse, rmse, r2 = evaluate_model(model, X_test, y_test)
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"Root Mean Squared Error: {rmse:.2f}")
    st.write(f"R^2 Score: {r2:.2f}")

    # Get user input for prediction
    st.header("Make a Prediction")
    # Use median values from the dataset as defaults.
    default_values = df.drop('Price', axis=1).median()

    # Create input widgets.  Make sure the order matches the columns in X_train.
    rooms = st.number_input("Rooms", value=int(default_values['Rooms']), step=1)
    type_input = st.selectbox("Type", options=df['Type'].unique(), index=0)  # Use index 0 as default
    distance = st.number_input("Distance", value=float(default_values['Distance']), step=0.1)
    postcode = st.number_input("Postcode", value=int(default_values['Postcode']), step=1)
    bedroom2 = st.number_input("Bedroom2", value=int(default_values['Bedroom2']), step=1)
    bathroom = st.number_input("Bathroom", value=float(default_values['Bathroom']), step=0.1)
    car = st.number_input("Car", value=int(default_values['Car']), step=1)
    landsize = st.number_input("Landsize", value=int(default_values['Landsize']), step=1)
    buildingarea = st.number_input("BuildingArea", value=float(default_values['BuildingArea']), step=0.1)
    yearbuilt = st.number_input("YearBuilt", value=int(default_values['YearBuilt']), step=1)
    councilarea = st.selectbox("CouncilArea", options=df['CouncilArea'].unique(), index=0)
    lattitude = st.number_input("Lattitude", value=float(default_values['Lattitude']), step=0.0001)
    longtitude = st.number_input("Longtitude", value=float(default_values['Longtitude']), step=0.0001)
    regionname = st.selectbox("Regionname", options=df['Regionname'].unique(), index=0)
    propertycount = st.number_input("Propertycount", value=int(default_values['Propertycount']), step=1)
    sellerg = st.selectbox("SellerG", options=df['SellerG'].unique(), index=0)
    method = st.selectbox("Method", options=df['Method'].unique(), index=0)
    suburb = st.selectbox("Suburb", options=df['Suburb'].unique(), index=0)


    # Make prediction button
    if st.button('Predict Price'):
        # Prepare input data as a DataFrame.  The order of columns is critical.
        input_data = pd.DataFrame({
            'Suburb': [suburb],
            'Rooms': [rooms],
            'Type': [type_input],
            'Method': [method],
            'SellerG': [sellerg],
            'Distance': [distance],
            'Postcode': [postcode],
            'Bedroom2': [bedroom2],
            'Bathroom': [bathroom],
            'Car': [car],
            'Landsize': [landsize],
            'BuildingArea': [buildingarea],
            'YearBuilt': [yearbuilt],
            'CouncilArea': [councilarea],
            'Lattitude': [lattitude],
            'Longtitude': [longtitude],
            'Regionname': [regionname],
            'Propertycount': [propertycount],
        })

        # Ensure correct column order
        input_data = input_data[X_test.columns]

        # Predict the price
        predicted_price = model.predict(input_data)[0]
        st.success(f'Predicted Price: ${predicted_price:.2f}')

if __name__ == '__main__':
    main()
