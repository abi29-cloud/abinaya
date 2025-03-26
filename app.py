import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from PIL import Image
import base64
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import tensorflow as tf

# Function to get base64 encoding of a file
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background of the Streamlit app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpeg;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown('<style>h1 { color: White ; }</style>', unsafe_allow_html=True)
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background/3.jfif')

# Streamlit app title
st.title("Project : Thief detection with deep learning")

# Define paths
model_path = "my_cnn_model.h5"  # Path to your trained CNN model

# Try loading the model with error handling
try:
    # Debugging: Check TensorFlow/Keras version
    st.write("TensorFlow version:", tf.__version__)
    st.write("Keras version:", tf.keras.__version__)

    # Load the model
    model = load_model(model_path)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise

# Function to preprocess an image for the model
def preprocess_image(img):
    # Resize the image to match the model's input size
    img = cv2.resize(img, (model.input_shape[1], model.input_shape[2]))

    # Normalize pixel values
    img = img / 255.0

    # Add an extra dimension for batch processing (even though it's a single image)
    img = img.reshape((1,) + img.shape)
    return img

# Streamlit file uploader for video input
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("File uploaded successfully!")

    # Open the video file
    cap = cv2.VideoCapture("temp_video.mp4")

    # Placeholder for the video frames and prediction result
    stframe = st.empty()
    prediction_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Can't receive frame (stream end?). Exiting...")
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_image(frame)

        # Make prediction using the model
        try:
            prediction = model.predict(preprocessed_frame)[0]  # Assuming model outputs a single value
            # Debugging: Check prediction result
            st.write(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            break

        # Process the prediction result (adjust based on your model's output)
        text = "⚠️ Thief Detected!" if prediction > 0.5 else "No Theft Detected"
        prediction_text.text(text)  # Display the prediction result at the bottom

        # Put text on the frame (adjust position and font size)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the frame to RGB (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame with prediction
        stframe.image(frame, channels='RGB')

    # Release resources after video ends
    cap.release()
