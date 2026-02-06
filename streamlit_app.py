import streamlit as st
import numpy as np
import cv2
from PIL import Image

# 1. SETUP & PAGE CONFIG
st.set_page_config(page_title="Microskill 3: Biomedical Image Analysis", layout="wide")

st.title("Microskill 3: Traditional Biomedical Image Analysis")
st.markdown("""
**Objective:** Explore edge detection and thresholding algorithms to understand how they handle noise and identify boundaries in biomedical images.
""")

# --- Sidebar: Navigation & Settings ---
st.sidebar.header("Navigation")
script_option = st.sidebar.radio(
    "Select Problem:", 
    ("Problem 3.1: Edge Detection", "Problem 3.2: Thresholding"),
    help="Navigate between the Edge Detection task and the Thresholding task."
)

st.sidebar.markdown("---")
st.sidebar.header("Global Settings")
mode = st.sidebar.radio(
    "Image Source:", 
    ("Synthetic Image", "Upload Image"),
    help="Start with the 'Synthetic Image' to replicate the homework problem, then try 'Upload Image' to test Rigor & Reproducibility."
)

# Shared Noise Slider (Crucial for Problem 3.1)
noise_level = st.sidebar.slider(
    "Noise Level", 0, 100, 50,
    help="Simulates sensor noise. See how Sobel fails and Canny succeeds as you increase this."
)

# --- Logic for Problem 3.1 (Edge Detection) ---
if script_option == "Problem 3.1: Edge Detection":
    st.header("Problem 3.1: Comparing Edge Detection Algorithms")
    
    # Instructions from Curriculum
    st.info("""
    ### Task:
    1. Adjust the settings below to apply Sobel and Canny detectors.
    2. Compare the outputs: Notice how Sobel picks up the 'static' (noise) while Canny filters it out.
    """)
    
    st.sidebar.subheader("Edge Parameters")
    kernel_size = st.sidebar.slider(
        "Sobel Kernel Size", 3, 11, 5, 2,
        help="Larger kernels average out some noise but make edges blurry. (Standard: 5)"
    )
    threshold1 = st.sidebar.slider(
        "Canny Threshold 1", 0, 255, 100,
        help="Lower bound. Edges weaker than this are rejected. Lowering this adds more detail (and noise)."
    )
    threshold2 = st.sidebar.slider(
        "Canny Threshold 2", 0, 255, 200,
        help="Upper bound. Gradients stronger than this are accepted as 'Sure Edges'."
    )

# --- Logic for Problem 3.2 (Thresholding) ---
elif script_option == "Problem 3.2: Thresholding":
    st.header("Problem 3.2: Thresholding for Binary Segmentation")
    
    # Instructions from Curriculum
    st.info("""
    ### Task:
    1. Apply Otsu's method to segment the foreground (rectangle) from the background.
    2. Adjust the **Blur Kernel** to see how smoothing is required before thresholding.
    """)

    st.sidebar.subheader("Thresholding Parameters")
    blur_kernel_size = st.sidebar.slider(
        "Blur Kernel Size", 1, 15, 5, 2,
        help="Gaussian Blur size. If this is 1 (no blur), Otsu's method might fail on noisy images."
    )

st.warning("RCR Reminder: Do not upload sensitive/patient data to public tools.")

# --- Image Generation / Upload Logic ---
img = None

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image (Testing Generalizability)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = np.array(image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        st.write("Waiting for upload...")
else:
    # Synthetic Image Generation (Matches Curriculum Code)
    img = np.zeros((300, 300), dtype=np.uint8) # Increased size slightly for visibility
    cv2.rectangle(img, (90, 90), (210, 210), 255, -1)
    noise = np.random.randint(0, noise_level + 1, (300, 300), dtype=np.uint8)
    img = cv2.add(img, noise)

# --- Processing & Visualization ---
if img is not None:
    
    # ---------------- PROBLEM 3.1 RENDER ----------------
    if script_option == "Problem 3.1: Edge Detection":
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)
        sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        canny_edges = cv2.Canny(img, threshold1, threshold2)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img, caption="Noisy Input", use_container_width=True)
        with col2:
            st.image(sobel_edges, caption="Sobel (Sensitive to Noise)", use_container_width=True)
        with col3:
            st.image(canny_edges, caption="Canny (Cleaner Edges)", use_container_width=True)

        st.markdown("""
        **Analysis:** As noted in the curriculum, Sobel is a simple gradient calculator, so it treats noise as edges. 
        Canny uses **non-maximum suppression** and **hysteresis** to keep lines thin and connected.
        """)

    # ---------------- PROBLEM 3.2 RENDER ----------------
    elif script_option == "Problem 3.2: Thresholding":
        # Handle case where blur kernel is 1 (creates error in GaussianBlur if not handled or just skips it)
        if blur_kernel_size > 1:
            blurred = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
        else:
            blurred = img
            
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Noisy Input", use_container_width=True)
        with col2:
            st.image(otsu_thresh, caption="Otsu's Thresholding", use_container_width=True)
            
        st.markdown("""
        **Analysis:** Otsu's method calculates the optimal separation point in the histogram. 
        Try reducing the **Blur Kernel Size** to 1 in the sidebar—you will likely see the background noise appear as "speckles" in the thresholded image.
        """)

# --- Rigor & RCR Footer ---
st.divider()
with st.expander("Review: Rigor, Reproducibility & Ethics"):
    st.markdown("""
    * **Algorithm Evaluation:** We test on synthetic data (squares) first. However, to ensure *generalizability*, we must also test on real biomedical data (MRI, CT).
    * **Ethics:** When using open-source libraries like OpenCV, proper citation is required to ensure transparency and credit.
    """)
