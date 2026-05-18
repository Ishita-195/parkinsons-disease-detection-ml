import streamlit as st
import pickle
import numpy as np

# Load model
with open("../models/parkinsons_svm.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("../models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.title("Parkinson's Disease Detection")

st.write("Enter biomedical voice measurements:")

feature_names = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE"
]

features = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    features.append(value)

if st.button("Predict"):

    input_data = np.array(
        features
    ).reshape(1, -1)

    input_scaled = scaler.transform(
        input_data
    )

    prediction = model.predict(
        input_scaled
    )[0]

    if prediction == 1:
        st.error("Parkinson's Detected")
    else:
        st.success("Healthy")