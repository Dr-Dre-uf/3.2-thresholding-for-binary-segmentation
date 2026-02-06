import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("Comparing Edge Detection Algoithms")

# Sidebar: select processing method
st.sidebar.header("Select Processing Method")
script_option = st.sidebar.radio("Choose a method:", ("Edge Detection", "Thresholding"))

# Shared options
mode = st.sidebar.radio("Image Source:", ("Synthetic Image", "Upload Image"))
noise_level = st.sidebar.slider("Noise Level", 0, 100, 50)

# If Edge Detection is selected ----
if script_option == "Edge Detection":
    st.sidebar.subheader("Edge Detection Settings")
    kernel_size = st.sidebar.slider("Sobel Kernel Size", 3, 11, 5, 2)
    threshold1 = st.sidebar.slider("Canny Threshold 1", 0, 255, 100)
    threshold2 = st.sidebar.slider("Canny Threshold 2", 0, 255, 200)

# If Thresholding is selected ----
elif script_option == "Thresholding":
    st.sidebar.subheader("Thresholding Settings")
    blur_kernel_size = st.sidebar.slider("Blur Kernel Size", 3, 15, 5, 2)

st.warning("Please do not upload any sensitive or personal data.")

# --- Image generation / upload ---
img = None

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = np.array(image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        st.info("Please upload an image.")
else:
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), 255, -1)
    noise = np.random.randint(0, noise_level, (100, 100), dtype=np.uint8)
    img = cv2.add(img, noise)

# --- Process and display image ---
if img is not None:

    # Edge Detection mode ----
    if script_option == "Edge Detection":
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)
        sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        canny_edges = cv2.Canny(img, threshold1, threshold2)

        st.header("Edge Detection Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img, caption="Original Image", use_column_width=True, clamp=True, channels="GRAY")
        with col2:
            st.image(sobel_edges, caption="Sobel Edges", use_column_width=True)
        with col3:
            st.image(canny_edges, caption="Canny Edges", use_column_width=True)

    # Thresholding mode ----
    elif script_option == "Thresholding":
        blurred = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        st.header("Thresholding Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_column_width=True, clamp=True, channels="GRAY")
        with col2:
            st.image(otsu_thresh, caption="Otsu Thresholded Image", use_column_width=True)
