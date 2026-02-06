# Microskill 3: Traditional Biomedical Image Analysis (Streamlit App)

This interactive application accompanies the **Microskill 3** curriculum. It provides a visual sandbox for students to explore traditional computer vision algorithms, specifically Edge Detection (Sobel vs. Canny) and Thresholding (Otsu's Method).

## Features

* **Problem 3.1: Edge Detection:** Compare the noise sensitivity of Sobel gradients vs. Canny edge detection.
* **Problem 3.2: Thresholding:** Experiment with Gaussian Blur pre-processing to see how it affects Otsu's binarization.
* **Rigor & Reproducibility:** Upload custom images to test if algorithms generalize beyond synthetic data.
* **Interactive Parameters:** Real-time sliders for kernel sizes, noise levels, and thresholds.

## Installation

1.  **Clone or download** this repository.
2.  (Optional) Create a virtual environment:
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On Mac/Linux:
    source venv/bin/activate
    ```
3.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit app locally with the following command:

```bash
streamlit run streamlit_app.py
