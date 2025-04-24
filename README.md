# CanaryEdgeDetection
# 🧠 Canny Edge Detection (Python Implementation)

This project demonstrates a manual implementation of the **Canny Edge Detection** algorithm in Python — without using OpenCV or other pre-built computer vision libraries. It walks through each step of the edge detection process and allows threshold tuning via sliders (if run in Google Colab) or static values (if run locally in VS Code).

---

## 📌 Features

- Full implementation of:
  - Grayscale conversion
  - Gaussian blur
  - Sobel filter (gradient magnitude & direction)
  - Non-maximum suppression
  - Double thresholding
  - Edge tracking by hysteresis
- Interactive threshold sliders in **Google Colab**
- Image upload support in both **Colab** and **local Python (VS Code)** setup
- Output includes:
  - Final edge-detected image (saved to disk)
  - Summary statistics (dimensions, intensity, pixel count)

---

## 🔧 Requirements

Install dependencies using:

```bash
pip install numpy matplotlib pillow ipywidgets
