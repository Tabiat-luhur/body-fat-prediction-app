import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('bodyfat.csv')
    return df

df = load_data()

# Separate features and target variable
X = df[['Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen']]
y = df['BodyFat']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Function to predict body fat percentage
def predict_body_fat(age, weight_kg, height_cm, neck_cm, chest_cm, abdomen_cm):
    # Convert height from cm to inches
    height_inches = height_cm / 2.54
    
    # Convert weight from kg to lbs
    weight_lbs = weight_kg / 0.453592
    
    input_data = np.array([[age, weight_lbs, height_inches, neck_cm, chest_cm, abdomen_cm]])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title('Body Fat Percentage Prediction App')
    st.write('Masukkan informasi berikut untuk memprediksi persentase body fat Anda.')

    # User inputs
    age = st.number_input('Usia (tahun)', min_value=18, max_value=99, value=30)
    weight_kg = st.number_input('Berat (kg)', min_value=36.29, max_value=181.44, value=68.0)
    height_cm = st.number_input('Tinggi (cm)', min_value=127.0, max_value=228.6, value=172.72)
    neck_cm = st.number_input('Lingkar Leher (cm)', min_value=30.0, max_value=60.0, value=35.0)
    chest_cm = st.number_input('Lingkar Dada (cm)', min_value=70.0, max_value=200.0, value=100.0)
    abdomen_cm = st.number_input('Lingkar Perut (cm)', min_value=60.0, max_value=150.0, value=85.0)

    # Prediction
    if st.button('Submit'):
        result = predict_body_fat(age, weight_kg, height_cm, neck_cm, chest_cm, abdomen_cm)
        st.write(f'Prediksi Persentase Body Fat: {result:.2f}%')

if __name__ == '__main__':
    main()
