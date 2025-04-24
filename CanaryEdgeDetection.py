# Upload your image from local system
from google.colab import files
uploaded = files.upload()

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
from ipywidgets import interact, FloatSlider
from PIL import Image
import os

# ------------------------
# 1. Convert RGB to Grayscale
# ------------------------
def rgb2gray(img):
    # Applies weighted average to convert RGB to grayscale
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

# ------------------------
# 2. Apply Gaussian Blur
# ------------------------
def gaussian_blur(img, kernel_size=5, sigma=1):
    # Generates a Gaussian kernel for smoothing
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    kernel /= np.sum(kernel)

    # Convolve kernel with the image
    pad = kernel_size // 2
    img_padded = np.pad(img, pad, mode='constant')
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = np.sum(kernel * img_padded[i:i+kernel_size, j:j+kernel_size])
    return result

# ------------------------
# 3. Generic Convolution Function
# ------------------------
def convolve(img, kernel):
    # Apply a kernel to an image using convolution
    k = kernel.shape[0] // 2
    padded = np.pad(img, ((k, k), (k, k)), mode='constant')
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            result[i, j] = np.sum(region * kernel)
    return result

# ------------------------
# 4. Sobel Filter for Gradient Magnitude & Direction
# ------------------------
def sobel_filters(img):
    # Detect edges using horizontal and vertical Sobel kernels
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Gx = convolve(img, Kx)
    Gy = convolve(img, Ky)

    # Combine into gradient magnitude and direction
    magnitude = np.hypot(Gx, Gy)
    direction = np.arctan2(Gy, Gx)

    return magnitude, direction

# ------------------------
# 5. Non-Maximum Suppression
# ------------------------
def non_maximum_suppression(magnitude, direction):
    # Suppress non-maximum pixels in the edge direction
    M, N = magnitude.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255

            # Based on angle, pick neighbors to compare
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                Z[i,j] = magnitude[i,j]
            else:
                Z[i,j] = 0
    return Z

# ------------------------
# 6. Double Thresholding
# ------------------------
def threshold(img, lowRatio, highRatio):
    # Set strong and weak edges based on intensity thresholds
    highThreshold = img.max() * highRatio
    lowThreshold = highThreshold * lowRatio

    res = np.zeros_like(img)
    weak = 25
    strong = 255

    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong

# ------------------------
# 7. Edge Tracking by Hysteresis
# ------------------------
def hysteresis(img, weak, strong=255):
    # Finalize edges: turn weak to strong if connected to strong
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if img[i,j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i,j] = strong
                else:
                    img[i,j] = 0
    return img

# ------------------------
# 8. Main Execution Logic with Sliders
# ------------------------
filename = list(uploaded.keys())[0]
img = mpimg.imread(filename)
gray = rgb2gray(img)
blurred = gaussian_blur(gray, 5, 1)
mag, direction = sobel_filters(blurred)
suppressed = non_maximum_suppression(mag, direction)

def run_canny(low, high):
    # Full Canny pipeline with given thresholds
    thresholded, weak, strong = threshold(suppressed, low, high)
    final = hysteresis(thresholded, weak, strong)

    # Show output
    plt.figure(figsize=(10, 5))
    plt.imshow(final, cmap='gray')
    plt.title(f"Edge Detection: Low={low:.2f}, High={high:.2f}")
    plt.axis('off')
    plt.show()

    # Save result
    im = Image.fromarray(final.astype(np.uint8))
    im.save("canny_output.png")
    print("✅ Output image saved as: canny_output.png")

    # Print image features
    print("\n📊 Image Summary:")
    print(f"Dimensions       : {final.shape}")
    print(f"Gray intensity   : min={gray.min():.2f}, max={gray.max():.2f}")
    print(f"Edges detected   : {(final > 0).sum()} pixels")

# ------------------------
# 9. Interactive Threshold Sliders
# ------------------------
interact(run_canny,
         low=FloatSlider(min=0.01, max=0.5, step=0.01, value=0.05, description='Low Threshold'),
         high=FloatSlider(min=0.05, max=1.0, step=0.01, value=0.15, description='High Threshold'));

