import streamlit as st
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path

st.set_page_config(page_title="Edge Detection Explorer", layout="wide")

st.title("Edge Detection Explorer: Sobel vs. Canny")

st.sidebar.header("Input")

sample_dir = Path("sample_images")
sample_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpeg"))

input_mode = st.sidebar.radio(
    "Choose input source",
    ["Upload image", "Use sample image"]
)

uploaded_file = None
selected_sample = None

if input_mode == "Upload image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"]
    )
else:
    if sample_files:
        selected_sample = st.sidebar.selectbox(
            "Select sample image",
            sample_files,
            format_func=lambda x: x.name
        )
    else:
        st.sidebar.warning("No sample images found in sample_images/.")

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

if input_mode == "Upload image":
    if uploaded_file is None:
        st.info("Please upload an image to get started.")
        st.stop()
    image = load_image(uploaded_file)

else:
    if selected_sample is None:
        st.info("Please add sample images to the sample_images folder.")
        st.stop()
    image = np.array(Image.open(selected_sample).convert("RGB"))
    
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

st.subheader("Gradient Histogram")

fig, ax = plt.subplots()
ax.hist(sobel_magnitude.ravel(), bins=50)
ax.axvline(sobel_threshold, linestyle="--", label="Sobel threshold")
ax.set_xlabel("Gradient magnitude")
ax.set_ylabel("Pixel count")
ax.set_title("Distribution of Sobel gradient values")
ax.legend()

st.pyplot(fig)

st.caption(
    "The histogram shows how many pixels have low or high gradient strength. "
    "The threshold line determines which pixels become Sobel edges."
)

st.subheader("Metrics")

sobel_edge_ratio = np.mean(sobel_edges > 0) * 100
canny_edge_ratio = np.mean(canny_edges > 0) * 100

sobel_components, sobel_num_components = ndimage.label(sobel_edges > 0)
canny_components, canny_num_components = ndimage.label(canny_edges > 0)

mean_gradient_on_sobel_edges = np.mean(sobel_magnitude[sobel_edges > 0]) if np.any(sobel_edges > 0) else 0
mean_gradient_on_canny_edges = np.mean(sobel_magnitude[canny_edges > 0]) if np.any(canny_edges > 0) else 0

col1, col2 = st.columns(2)

with col1:
    st.metric("Sobel edge pixels", f"{sobel_edge_ratio:.2f}%")
    st.metric("Sobel components", sobel_num_components)
    st.metric("Mean gradient on Sobel edges", f"{mean_gradient_on_sobel_edges:.1f}")

with col2:
    st.metric("Canny edge pixels", f"{canny_edge_ratio:.2f}%")
    st.metric("Canny components", canny_num_components)
    st.metric("Mean gradient on Canny edges", f"{mean_gradient_on_canny_edges:.1f}")


st.subheader("Interpretation")
st.write(
    "Sobel highlights intensity changes and requires a threshold to create a binary edge map. "
    "Canny uses a stronger edge detection pipeline and typically produces thinner, cleaner edges."
)