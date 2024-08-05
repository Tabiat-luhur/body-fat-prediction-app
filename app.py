import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('bodyfat.csv')
    return df

df = load_data()

# Separate features and target variable
X = df[['Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist']]
y = df['BodyFat']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipelines for each model with StandardScaler
pipelines = [
    ('rf', Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=2, bootstrap=True))])),
    ('gbr', Pipeline([('scaler', StandardScaler()), ('model', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3))])),
    ('svr', Pipeline([('scaler', StandardScaler()), ('model', SVR(C=1, gamma='scale', kernel='rbf'))]))
]

# Define the stacking model
stacking_model = StackingRegressor(estimators=pipelines, final_estimator=LinearRegression())

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Function to predict body fat percentage
def predict_body_fat(age, weight_kg, height_cm, neck_cm, chest_cm, abdomen_cm, hip_cm, thigh_cm, knee_cm, ankle_cm, biceps_cm, forearm_cm, wrist_cm):
    # Convert height from cm to inches
    height_inches = height_cm / 2.54
    # Convert weight from kg to lbs
    weight_lbs = weight_kg / 0.453592
    input_data = np.array([[age, weight_lbs, height_inches, neck_cm, chest_cm, abdomen_cm, hip_cm, thigh_cm, knee_cm, ankle_cm, biceps_cm, forearm_cm, wrist_cm]])
    prediction = stacking_model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title('Body Fat Percentage Prediction App')
    st.write('Masukkan informasi berikut untuk memprediksi persentase body fat Anda.')

    with st.form(key='predict_form'):
        age = st.number_input('Usia (tahun)', min_value=18, max_value=99, value=30)
        weight_kg = st.number_input('Berat (kg)', min_value=36.29, max_value=181.44, value=68.0)
        height_cm = st.number_input('Tinggi (cm)', min_value=127.0, max_value=228.6, value=172.72)
        neck_cm = st.number_input('Lingkar Leher (cm)', min_value=30.0, max_value=60.0, value=35.0)
        chest_cm = st.number_input('Lingkar Dada (cm)', min_value=70.0, max_value=200.0, value=100.0)
        abdomen_cm = st.number_input('Lingkar Perut (cm)', min_value=60.0, max_value=150.0, value=85.0)
        hip_cm = st.number_input('Lingkar Pinggul (cm)', min_value=70.0, max_value=150.0, value=95.0)
        thigh_cm = st.number_input('Lingkar Paha (cm)', min_value=30.0, max_value=100.0, value=55.0)
        knee_cm = st.number_input('Lingkar Lutut (cm)', min_value=20.0, max_value=70.0, value=40.0)
        ankle_cm = st.number_input('Lingkar Pergelangan Kaki (cm)', min_value=15.0, max_value=40.0, value=23.0)
        biceps_cm = st.number_input('Lingkar Biceps (cm)', min_value=20.0, max_value=60.0, value=30.0)
        forearm_cm = st.number_input('Lingkar Lengan Bawah (cm)', min_value=20.0, max_value=40.0, value=25.0)
        wrist_cm = st.number_input('Lingkar Pergelangan Tangan (cm)', min_value=10.0, max_value=25.0, value=15.0)

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        result = predict_body_fat(age, weight_kg, height_cm, neck_cm, chest_cm, abdomen_cm, hip_cm, thigh_cm, knee_cm, ankle_cm, biceps_cm, forearm_cm, wrist_cm)
        st.write(f'Prediksi Persentase Body Fat: {result:.2f}%')

if __name__ == '__main__':
    main()
