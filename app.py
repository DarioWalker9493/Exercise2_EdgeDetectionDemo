import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Edge Detection Explorer", layout="wide")

st.title("Edge Detection Explorer: Sobel vs. Canny")

st.sidebar.header("Input")

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

def load_image(file):
    image = Image.open(file).convert("RGB")
    return np.array(image)

if uploaded_file is not None:
    image = load_image(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)
else:
    st.info("Please upload an image to get started.")