import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

# 1. Page Configuration
st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title(" Draw a Digit (0-9)")
st.write("Draw a number in the box below, and the neural network will try to guess it!")

# 2. Load the Model 
# We use st.cache_resource so the model only loads once, preventing lag.
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_robust_cnn.keras')

model = load_model()

# 3. Create the Drawing Canvas
# MNIST images are white digits on a black background.
canvas_result = st_canvas(
    fill_color="black",  # Background color
    stroke_width=15,     # Thickness of the drawing pen
    stroke_color="white",# Pen color
    background_color="black",
    width=280,           # Canvas width
    height=280,          # Canvas height
    drawing_mode="freedraw",
    key="canvas",
)

# 4. Process the Image and Predict
if canvas_result.image_data is not None:
    # A button to trigger the prediction
    if st.button("Predict Digit"):
        # The canvas outputs an RGBA array (Height, Width, 4). 
        # We only need the grayscale values.
        img = canvas_result.image_data
        
        # Convert RGBA to Grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        
        # Resize from 280x280 down to 28x28 to match the MNIST training data
        resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize the pixel values (0 to 1) just like we did in the notebook
        normalized_image = resized_image / 255.0
        
        # Reshape to match the model's expected input shape: (batch_size, height, width)
        input_image = normalized_image.reshape(1, 28, 28)
        
        # Make the prediction
        prediction = model.predict(input_image)
        guessed_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Display the results
        st.success(f"### Prediction: **{guessed_digit}**")
        st.info(f"Confidence: {confidence:.2f}%")
        
        # Optional: Show what the model actually "saw" after resizing
        st.write("Here is the 28x28 image the model processed:")
        st.image(resized_image, width=150)