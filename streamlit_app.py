import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Image Edge & Threshold Lab", layout="wide")

st.title("Comparing Edge Detection Algorithms")
st.markdown("""
This tool allows you to explore how different computer vision algorithms identify boundaries and segments in an image. 
**Select a method in the sidebar** and adjust the sliders to see real-time results.
""")

# --- Sidebar: select processing method ---
st.sidebar.header("Navigation")
script_option = st.sidebar.radio(
    "Choose a method:", 
    ("Edge Detection", "Thresholding"),
    help="Edge Detection finds boundaries; Thresholding separates foreground from background."
)

# Shared options
st.sidebar.markdown("---")
st.sidebar.header("Global Settings")
mode = st.sidebar.radio(
    "Image Source:", 
    ("Synthetic Image", "Upload Image"),
    help="Use a clean generated square or upload your own file."
)

noise_level = st.sidebar.slider(
    "Noise Level", 0, 100, 50,
    help="Higher values add random variation to the image, making it harder for algorithms to find clean edges."
)

# --- Algorithm Specific Settings ---
if script_option == "Edge Detection":
    st.sidebar.subheader("Edge Detection Settings")
    
    kernel_size = st.sidebar.slider(
        "Sobel Kernel Size", 3, 11, 5, 2,
        help="The size of the 'window' used to calculate gradients. Larger kernels are more robust to noise but less precise."
    )
    
    threshold1 = st.sidebar.slider(
        "Canny Threshold 1", 0, 255, 100,
        help="The lower bound for hysteresis thresholding. Edges below this are discarded."
    )
    
    threshold2 = st.sidebar.slider(
        "Canny Threshold 2", 0, 255, 200,
        help="The upper bound. Any gradient higher than this is definitely an edge."
    )

elif script_option == "Thresholding":
    st.sidebar.subheader("Thresholding Settings")
    blur_kernel_size = st.sidebar.slider(
        "Blur Kernel Size", 3, 15, 5, 2,
        help="Gaussian blur helps reduce noise before applying Otsu's thresholding. Must be an odd number."
    )

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
        st.info("Please upload an image to begin.")
else:
    # Generate synthetic image
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(img, (75, 75), (225, 225), 255, -1)
    noise = np.random.randint(0, noise_level + 1, (300, 300), dtype=np.uint8)
    img = cv2.add(img, noise)

# --- Process and display image ---
if img is not None:
    if script_option == "Edge Detection":
        # Sobel
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)
        sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Canny
        canny_edges = cv2.Canny(img, threshold1, threshold2)

        st.header("Edge Detection Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img, caption="Original (Grayscale)", use_container_width=True)
        with col2:
            st.image(sobel_edges, caption="Sobel (Gradient Magnitude)", use_container_width=True)
        with col3:
            st.image(canny_edges, caption="Canny (Refined Edges)", use_container_width=True)
            
        st.info("**Tip:** If Canny is too messy, try increasing Threshold 2. If it's missing edges, lower Threshold 1.")

    elif script_option == "Thresholding":
        blurred = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        st.header("Otsu's Binarization Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(otsu_thresh, caption="Otsu Thresholded", use_container_width=True)
        
        st.info("**How it works:** Otsu's method automatically finds the optimal threshold to separate the background and foreground pixels based on the image histogram.")
