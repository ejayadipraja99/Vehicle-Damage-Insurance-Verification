import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Assuming you have a list of label names corresponding to your model output
label_names = ["crack", "scratch", "tire flat", "dent", "glass shatter", "lamp broken"]

def run():
    # Load the saved model
    model = tf.keras.models.load_model("best_model.h5")
    
    # Allow the user to select an image file
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the image using PIL
        img = Image.open(uploaded_file)
        
        # Preprocess the image
        image_size = (150, 150)
        img = img.resize(image_size)

        # Convert the PIL.Image.Image object to a NumPy array
        x = np.array(img)

        # Expand the array to add a batch dimension
        x = np.expand_dims(x, axis=0)

        # Normalize the image data
        x = x / 255.0

        # Make the prediction using the loaded model
        y_pred = model.predict(x)

        # Get the index of the predicted class with the highest probability
        class_idx = np.argmax(y_pred, axis=1)[0]

        # Display the predicted class label and image to the user
        st.write(f"Predicted Class: {label_names[class_idx]}")
        st.image(img, caption=f"Predicted Class: {label_names[class_idx]}", use_column_width=True)

if __name__ == "__main__":
    run()
