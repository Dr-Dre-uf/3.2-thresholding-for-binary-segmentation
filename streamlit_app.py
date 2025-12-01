import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

st.title("Image Processing App")

# Main selection sidebar

st.sidebar.header("Select Processing Method")
script_option = st.sidebar.radio(
"Choose a method:",
("Edge Detection", "Thresholding")
)

# Shared options

mode = st.sidebar.radio("Image Source:", ("Synthetic Image", "Upload Image"))
noise_level = st.sidebar.slider("Noise Level", 0, 100, 50, help="Amount of random noise added to the image.")

# Edge Detection specific options

if script_option == "Edge Detection":
st.sidebar.subheader("Edge Detection Settings")
kernel_size = st.sidebar.slider("Sobel Kernel Size", 3, 11, 5, 2, help="Size of Sobel kernel")
threshold1 = st.sidebar.slider("Canny Threshold 1", 0, 255, 100)
threshold2 = st.sidebar.slider("Canny Threshold 2", 0, 255, 200)
colormap = st.sidebar.selectbox("Colormap", ["gray", "viridis", "plasma", "magma", "inferno"])

# Thresholding specific options

elif script_option == "Thresholding":
st.sidebar.subheader("Thresholding Settings")
blur_kernel_size = st.sidebar.slider("Blur Kernel Size", 3, 15, 5, 2, help="Gaussian blur kernel size")

# Warning

st.warning("Please do not upload any sensitive or personal data.")

# Image Generation / Upload

img = None
if mode == "Upload Image":
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
image = Image.open(uploaded_file)
img = np.array(image)
if len(img.shape) == 3:
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
st.info("Please upload an image to continue.")
else:
img = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(img, (30, 30), (70, 70), 255, -1)
noise = np.random.randint(0, noise_level, (100, 100), dtype=np.uint8)
img = cv2.add(img, noise)

# Process image based on selection

if img is not None:
if script_option == "Edge Detection":
# Sobel and Canny
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
sobel_edges = cv2.magnitude(sobel_x, sobel_y)
canny_edges = cv2.Canny(img, threshold1, threshold2)

```
    # Normalize
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    canny_edges = cv2.normalize(canny_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Display
    st.header("Edge Detection Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.imshow(img, cmap=colormap)
        ax1.axis('off')
        ax1.set_title("Original Image")
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.imshow(sobel_edges, cmap=colormap)
        ax2.axis('off')
        ax2.set_title("Sobel Edges")
        st.pyplot(fig2)
    with col3:
        fig3, ax3 = plt.subplots()
        ax3.imshow(canny_edges, cmap=colormap)
        ax3.axis('off')
        ax3.set_title("Canny Edges")
        st.pyplot(fig3)

elif script_option == "Thresholding":
    # Gaussian Blur + Otsu
    blurred = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display
    st.header("Thresholding Results")
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
```
