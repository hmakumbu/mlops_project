import streamlit as st
import requests
from PIL import Image
import os

# Define the FastAPI backend URL (adjust this based on where your API is hosted)
API_URL = "http://localhost:8000"  # Change this if hosted elsewhere

# Token placeholder for authenticated requests
token = None

# Function to login and get JWT token
def login(username, password):
    response = requests.post(f"{API_URL}/token", data={"username": username, "password": password})
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        st.error("Login failed! Please check your username and password.")
        return None

# Function to show predicted segmentations (POST request)
def show_predicted_segmentations(samples_list, slice_to_plot):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_URL}/showPredictSegmented/", json={"samples_list": samples_list, "slice_to_plot": slice_to_plot}, headers=headers)
    if response.status_code == 200:
        st.success(response.json().get("message"))
    else:
        st.error("Error in fetching predicted segmentations.")

# Function to evaluate the model
def evaluate_model():
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_URL}/evaluate", headers=headers)
    if response.status_code == 200:
        metrics = response.json()
        for key, value in metrics.items():
            st.write(f"{key}: {value}")
    else:
        st.error("Error in evaluating the model.")

# Function to predict brain segmentation
def predict_segmentation(case_path, case):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_URL}/predict/", json={"case_path": case_path, "case": case}, headers=headers)
    if response.status_code == 200:
        st.success("Prediction successful!")
        prediction = response.json().get("prediction")
        st.write(f"Prediction: {prediction}")
    else:
        st.error("Error in making the prediction.")

# Function to show drift (placeholder)
def show_drift():
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_URL}/showdrift/", headers=headers)
    if response.status_code == 200:
        st.success(response.json().get("message"))
    else:
        st.error("Error in fetching drift status.")

# Streamlit UI components

st.title("Brain Segmentation and Evaluation Dashboard")

# Login section
st.header("Login to Access the API")
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    token = login(username, password)
    if token:
        st.success("Login successful!")
    else:
        st.error("Login failed!")

# Segmentation section
if token:
    st.header("Predict Brain Segmentation")
    case_path = st.text_input("Case Path (e.g., /path/to/case)")
    case = st.text_input("Case ID (e.g., BraTS2020)")
    if st.button("Predict Segmentation"):
        predict_segmentation(case_path, case)

# Evaluation section
if token:
    st.header("Evaluate the Model")
    if st.button("Evaluate Model"):
        evaluate_model()

# Show Predicted Segmentations section
if token:
    st.header("Show Predicted Segmentations")
    samples_list = st.text_input("Samples List (comma separated, e.g., sample1,sample2)").split(",")
    slice_to_plot = st.number_input("Slice to Plot", min_value=0, max_value=100, value=60)
    if st.button("Show Predicted Segmentations"):
        show_predicted_segmentations(samples_list, slice_to_plot)

# Show Drift section
if token:
    st.header("Show Drift Status")
    if st.button("Check Drift"):
        show_drift()