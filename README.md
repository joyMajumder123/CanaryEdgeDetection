
# Canny Edge Detection (Python Implementation)

**Context**

Digital images captured in real-world environments often contain multiple colors, varying illumination, and background clutter. To analyze objects in such images, it is important to detect their boundaries accurately. Edge detection helps in identifying object outlines for further image processing tasks.

**Problem**

Simple edge detection techniques are sensitive to noise and may either miss important edges or detect false ones. Variations in lighting and color also make it difficult to obtain clear and continuous object boundaries, leading to inaccurate results in image analysis.

**Solution**

Canny Edge Detection provides a multi-stage approach that includes noise reduction, gradient calculation, non-maximum suppression, and thresholding. This method effectively detects strong and continuous edges while minimizing noise, resulting in accurate boundary detection.


## Features

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

##  Requirements

Install dependencies using:

`bash
pip install numpy matplotlib pillow ipywidgets

Here is a working demo in google collab --  https://colab.research.google.com/drive/1UlR7KkHPBldshUSsq0YBGJruSU9Am9hq?usp=sharing
<img width="459" height="494" alt="image" src="https://github.com/user-attachments/assets/40f3696d-4342-4879-bd91-57b7a7ad905f" />

