# Edge Detection Explorer: Sobel vs. Canny

## Project Summary

This project is an interactive Streamlit app that demonstrates and compares two fundamental edge detection methods in image processing:

- **Sobel Operator** – computes image gradients to highlight intensity changes
- **Canny Edge Detector** – a multi-stage algorithm that produces clean, thin edges

The app allows users to explore how parameters such as smoothing and thresholds influence the results and to understand the trade-offs between noise sensitivity and edge quality.

---

## Objective

The goal of this app is to help users:

- Understand how edge detection works
- Compare gradient-based vs. pipeline-based methods
- Explore how parameters affect results
- Develop intuition for noise vs. detail trade-offs

---

## Features

### Input
- Upload your own image
- Select from built-in sample images

### Interactive Controls
- Gaussian blur (preprocessing)
- Sobel threshold
- Canny low threshold
- Canny high threshold

### Visualization
- Side-by-side comparison:
  - Original image
  - Sobel gradient / thresholded edges
  - Canny edges
- Gradient heatmap
- Histogram of gradient values

### Metrics
- Edge pixel ratio (%)
- Number of connected components
- Mean gradient magnitude

### Interpretation
- Explanation of parameter effects
- Guidance on trade-offs between noise and edge detection quality

---

## Installation (Local Setup)

### 1. Clone the repository

```bash
git clone https://github.com/DarioWalker9493/Exercise2_EdgeDetectionDemo
cd Excercise2_EdgeDetectionDemo
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the environment (Windows)
```bash
venv\Scripts\activate
```


### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
streamlit run app.py
```

## Limitations

- Sobel is sensitive to noise and may detect unwanted texture as edges.
- Canny edge detection strongly depends on threshold selection.
- The app currently works only on grayscale edge detection internally.
- Performance may decrease for very large images.
- The metrics are simplified and intended for educational comparison rather than rigorous benchmarking.
- Only Sobel and Canny methods are implemented; more advanced approaches such as Laplacian of Gaussian or deep learning-based edge detection are not included.

## Future Improvements

Possible future extensions include:

- Additional edge detection algorithms
- Color edge detection
- Automatic threshold optimization

## Author
Dario Walker
dario.walker@students.unibe.ch