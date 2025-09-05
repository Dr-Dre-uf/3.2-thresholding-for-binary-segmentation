import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

st.title("Otsu Thresholding for Binary Segmentation")

# Warning message
st.warning("Please do not upload any sensitive or personal data.")

# Sidebar for interactivity
st.sidebar.header("Image Parameters")
mode = st.sidebar.radio(
    "Choose a mode:",
    ("Synthetic Image", "Upload Image")
)

noise_level = st.sidebar.slider("Noise Level", 0, 100, 50, help="Controls the amount of random noise added to the image.")
blur_kernel_size = st.sidebar.slider("Blur Kernel Size", 3, 15, 5, 2, help="Size of the Gaussian blur kernel.")

# Image Generation/Upload
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], help="Upload an image to process.")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = np.array(image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        st.info("Please upload an image to continue.")
        img = None
else:
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), 255, -1)
    noise = np.random.randint(0, noise_level, (100, 100), dtype=np.uint8)
    img = cv2.add(img, noise)

if img is not None:
    # Apply Otsu's thresholding
    blurred = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display the results
    st.header("Thresholding Results")
    st.markdown("Adjust the noise level and blur kernel size to see their effect on the thresholding result.")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.imshow(img, cmap='gray')
        ax1.axis('off')
        ax1.set_title("Original Image")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.imshow(otsu_thresh, cmap='gray')
        ax2.axis('off')
        ax2.set_title("Otsu Thresholded Image")
        st.pyplot(fig2)