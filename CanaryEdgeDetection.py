import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg

# Converts an RGB image to grayscale
# Input: img (numpy array) - RGB image with shape (height, width, 3)
# Output: grayscale image (numpy array) with shape (height, width)
def rgb2gray(img):
    # Applies weighted sum of RGB channels using standard luminance coefficients
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

# Applies Gaussian blur to reduce noise in the image
# Input: img (numpy array) - Grayscale image
#        kernel_size (int) - Size of the Gaussian kernel (default: 5)
#        sigma (float) - Standard deviation of the Gaussian (default: 1)
# Output: blurred image (numpy array) with same shape as input
def gaussian_blur(img, kernel_size=5, sigma=1):
    # Create 1D Gaussian kernel using linspace for evenly spaced points
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    # Compute Gaussian values for the kernel
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    # Create 2D kernel by outer product of 1D Gaussian
    kernel = np.outer(gauss, gauss)
    # Normalize kernel to ensure sum equals 1
    kernel /= np.sum(kernel)

    # Calculate padding size for convolution
    pad = kernel_size // 2
    # Pad image to handle border pixels
    img_padded = np.pad(img, pad, mode='constant')
    # Initialize output array with same shape as input
    result = np.zeros_like(img)

    # Iterate over each pixel in the image using while loops
    i = 0
    while i < img.shape[0]:
        j = 0
        while j < img.shape[1]:
            # Apply convolution: sum of element-wise product of kernel and image region
            result[i, j] = np.sum(kernel * img_padded[i:i+kernel_size, j:j+kernel_size])
            j += 1
        i += 1
    return result

# Performs convolution of an image with a kernel
# Input: img (numpy array) - Input image (grayscale)
#        kernel (numpy array) - Convolution kernel (e.g., 3x3)
# Output: convolved image (numpy array) with same shape as input
def convolve(img, kernel):
    # Calculate half the kernel size for padding
    k = kernel.shape[0] // 2
    # Pad image to handle borders during convolution
    padded = np.pad(img, ((k, k), (k, k)), mode='constant')
    # Initialize output array
    result = np.zeros_like(img)

    # Iterate over each pixel in the image
    i = 0
    while i < img.shape[0]:
        j = 0
        while j < img.shape[1]:
            # Extract region of image corresponding to kernel size
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            # Compute convolution: sum of element-wise product of region and kernel
            result[i, j] = np.sum(region * kernel)
            j += 1
        i += 1
    return result

# Applies Sobel filters to compute gradient magnitude and direction
# Input: img (numpy array) - Grayscale image
# Output: magnitude (numpy array) - Gradient magnitude at each pixel
#         direction (numpy array) - Gradient direction in radians
def sobel_filters(img):
    # Define Sobel kernels for x and y directions
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # Horizontal gradient
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Vertical gradient

    # Apply convolution with Sobel kernels
    Gx = convolve(img, Kx)  # Gradient in x-direction
    Gy = convolve(img, Ky)  # Gradient in y-direction

    # Compute gradient magnitude using Euclidean norm
    magnitude = np.hypot(Gx, Gy)
    # Compute gradient direction using arctangent
    direction = np.arctan2(Gy, Gx)

    return magnitude, direction

# Performs non-maximum suppression to thin edges
# Input: magnitude (numpy array) - Gradient magnitude
#        direction (numpy array) - Gradient direction in radians
# Output: suppressed image (numpy array) - Thinned edges
def non_maximum_suppression(magnitude, direction):
    # Get image dimensions
    M, N = magnitude.shape
    # Initialize output array for suppressed edges
    Z = np.zeros((M, N), dtype=np.float32)
    # Convert direction to degrees for easier comparison
    angle = direction * 180. / np.pi
    # Normalize negative angles to [0, 180]
    angle[angle < 0] += 180

    # Iterate over inner pixels (excluding borders)
    i = 1
    while i < M-1:
        j = 1
        while j < N-1:
            # Initialize neighbor values for comparison
            q = 255
            r = 255
            # Check gradient direction and select appropriate neighbors
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                # Horizontal direction (0° or 180°)
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                # Diagonal (45°)
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                # Vertical (90°)
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                # Diagonal (135°)
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            # Keep pixel if its magnitude is greater than or equal to neighbors
            if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                Z[i,j] = magnitude[i,j]
            else:
                Z[i,j] = 0
            j += 1
        i += 1
    return Z

# Applies double thresholding to classify edges
# Input: img (numpy array) - Image after non-maximum suppression
#        lowRatio (float) - Ratio for low threshold (default: 0.05)
#        highRatio (float) - Ratio for high threshold (default: 0.15)
# Output: res (numpy array) - Thresholded image with strong/weak edges
#         weak (int) - Value for weak edges
#         strong (int) - Value for strong edges
def threshold(img, lowRatio=0.05, highRatio=0.15):
    # Calculate thresholds based on maximum gradient magnitude
    highThreshold = img.max() * highRatio
    lowThreshold = highThreshold * lowRatio

    # Initialize output array
    res = np.zeros_like(img)
    # Define values for weak and strong edges
    weak = 25
    strong = 255

    # Identify strong and weak edge pixels
    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    # Assign strong and weak values to corresponding pixels
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong

# Performs hysteresis to finalize edge detection
# Input: img (numpy array) - Thresholded image with strong/weak edges
#        weak (int) - Value for weak edges
#        strong (int) - Value for strong edges (default: 255)
# Output: final image (numpy array) - Edges after hysteresis
def hysteresis(img, weak, strong=255):
    # Get image dimensions
    M, N = img.shape
    # Iterate over inner pixels (excluding borders)
    i = 1
    while i < M-1:
        j = 1
        while j < N-1:
            # Check if pixel is a weak edge
            if img[i,j] == weak:
                # If any neighboring pixel is strong, mark this pixel as strong
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i,j] = strong
                else:
                    # If no strong neighbors, suppress the pixel
                    img[i,j] = 0
            j += 1
        i += 1
    return img

# Performs complete Canny edge detection pipeline
# Input: image_path (str) - Path to the input image
# Output: final_img (numpy array) - Detected edges
def canny_edge_detection(image_path):
    # Read the input image
    img = mpimg.imread(image_path)
    # Convert to grayscale
    gray = rgb2gray(img)
    # Apply Gaussian blur to reduce noise
    blurred = gaussian_blur(gray, 5, 1)
    # Compute gradient magnitude and direction using Sobel filters
    mag, direction = sobel_filters(blurred)
    # Apply non-maximum suppression to thin edges
    suppressed = non_maximum_suppression(mag, direction)
    # Apply double thresholding to classify edges
    thresholded, weak, strong = threshold(suppressed)
    # Apply hysteresis to finalize edges
    final_img = hysteresis(thresholded, weak, strong)

    return final_img

# Main execution block
# Selects the first uploaded image and runs edge detection
import os
# Iterate over dictionary keys to get the first key (filename)
for key in uploaded.keys():
    filename = key
    break

# Run Canny edge detection on the selected image
edges = canny_edge_detection(filename)

# Display the result
plt.figure(figsize=(10, 5))  # Create a figure for plotting
plt.imshow(edges, cmap='gray')  # Display edges in grayscale
plt.title("Canny Edge Detection (Manual)")  # Set title
plt.axis('off')  # Hide axes
plt.show()  # Show the plot
