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

if uploaded_file is None:
    st.info("Please upload an image to get started.")
    st.stop()

image = load_image(uploaded_file)
gray, sobel_magnitude, sobel_edges = compute_sobel_edges(image, sobel_threshold)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

with col2:
    st.subheader("Sobel Gradient Magnitude")
    st.image(sobel_magnitude, use_container_width=True, clamp=True)

with col3:
    st.subheader("Thresholded Sobel Edges")
    st.image(sobel_edges, use_container_width=True, clamp=True)

st.subheader("Interpretation")
st.write(
    "Sobel highlights regions where image intensity changes strongly. "
    "Increasing the threshold keeps only stronger edges, but weak details may disappear."
)