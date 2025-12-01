import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

st.title("Image Processing App")

# Main selection sidebar

st.sidebar.header("Select Script")
script_option = st.sidebar.radio(
"Choose a processing method:",
("Edge Detection", "Thresholding")
)

# Shared sidebar options

mode = st.sidebar.radio("Choose a mode:", ("Synthetic Image", "Upload Image"))

# Edge Detection specific options

if script_option == "Edge Detection":
st.sidebar.header("Edge Detection Parameters")
noise_level = st.sidebar.slider("Noise Level", 0, 100, 50)
kernel_size = st.sidebar.slider("Sobel Kernel Size", 3, 11, 5, 2)
threshold1 = st.sidebar.slider("Canny Threshold 1", 0, 255, 100)
threshold2 = st.sidebar.slider("Canny Threshold 2", 0, 255, 200)
colormap = st.sidebar.selectbox("Colormap", ["gray", "viridis", "plasma", "magma", "inferno"])

# Thresholding specific options

elif script_option == "Thresholding":
st.sidebar.header("Thresholding Parameters")
noise_level = st.sidebar.slider("Noise Level", 0, 100, 50)
blur_kernel_size = st.sidebar.slider("Blur Kernel Size", 3, 15, 5, 2)

# Warning

st.warning("Please do not upload any sensitive or personal data.")

# Image Generation / Upload

uploaded_file = None
if mode == "Upload Image":
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
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

# Run selected script

if img is not None:
if script_option == "Edge Detection":
# Apply Edge Detection
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
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.imshow(sobel_edges, cmap=colormap)
        ax2.axis('off')
        st.pyplot(fig2)
    with col3:
        fig3, ax3 = plt.subplots()
        ax3.imshow(canny_edges, cmap=colormap)
        ax3.axis('off')
        st.pyplot(fig3)

elif script_option == "Thresholding":
    # Apply Thresholding
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
