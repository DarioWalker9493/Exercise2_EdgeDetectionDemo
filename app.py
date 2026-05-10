import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Edge Detection Explorer", layout="wide")

st.title("Edge Detection Explorer: Sobel vs. Canny")

st.sidebar.header("Input")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

st.sidebar.header("Sobel Controls")
sobel_threshold = st.sidebar.slider("Sobel Threshold", 0, 255, 100)

st.sidebar.header("Canny Controls")
canny_low = st.sidebar.slider("Canny Low Threshold", 0, 255, 50)
canny_high = st.sidebar.slider("Canny High Threshold", 0, 255, 150)

st.sidebar.header("Preprocessing")
sigma = st.sidebar.slider("Gaussian Blur (sigma)", 0.0, 5.0, 1.0)

def load_image(file):
    image = Image.open(file).convert("RGB")
    return np.array(image)

def compute_sobel_edges(image_rgb, threshold):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    if magnitude.max() > 0:
        magnitude = (255 * magnitude / magnitude.max()).astype(np.uint8)
    else:
        magnitude = magnitude.astype(np.uint8)

    _, sobel_edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)

    return gray, magnitude, sobel_edges

def compute_canny_edges(gray, low_threshold, high_threshold):
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

if uploaded_file is None:
    st.info("Please upload an image to get started.")
    st.stop()

image = load_image(uploaded_file)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

if sigma > 0:
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
else:
    blurred = gray

sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

if sobel_magnitude.max() > 0:
    sobel_magnitude = (255 * sobel_magnitude / sobel_magnitude.max()).astype(np.uint8)
else:
    sobel_magnitude = sobel_magnitude.astype(np.uint8)

_, sobel_edges = cv2.threshold(sobel_magnitude, sobel_threshold, 255, cv2.THRESH_BINARY)

canny_edges = cv2.Canny(blurred, canny_low, canny_high)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

with col2:
    st.subheader("Sobel Edges")
    st.image(sobel_edges, use_container_width=True, clamp=True)

with col3:
    st.subheader("Canny Edges")
    st.image(canny_edges, use_container_width=True, clamp=True)

st.subheader("Diagnostic View")
st.image(sobel_magnitude, caption="Sobel Gradient Magnitude", use_container_width=True, clamp=True)

st.subheader("Interpretation")
st.write(
    "Sobel highlights intensity changes and requires a threshold to create a binary edge map. "
    "Canny uses a stronger edge detection pipeline and typically produces thinner, cleaner edges."
)