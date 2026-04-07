import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from google.colab import files

from google.colab import files
uploaded = files.upload()

# Converts RGB image to grayscale
def rgb2gray(img):
  return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

# Gaussian blur
def gaussian_blur(img, kernel_size=5, sigma=1):
  ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
  gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
  kernel = np.outer(gauss, gauss)
  kernel /= np.sum(kernel)
  pad = kernel_size // 2
  img_padded = np.pad(img, pad, mode='constant')
  result = np.zeros_like(img)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      region = img_padded[i:i+kernel_size, j:j+kernel_size]
      result[i, j] = np.sum(kernel * region)
  return result

# Convolution
def convolve(img, kernel):
  k = kernel.shape[0] // 2
  padded = np.pad(img, ((k, k), (k, k)), mode='constant')
  result = np.zeros_like(img)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
      result[i, j] = np.sum(region * kernel)
  return result

# Sobel filters
def sobel_filters(img):
  Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
  Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
  Gx = convolve(img, Kx)
  Gy = convolve(img, Ky)
  magnitude = np.hypot(Gx, Gy)
  direction = np.arctan2(Gy, Gx)
  return magnitude, direction

# Non-maximum suppression
def non_maximum_suppression(magnitude, direction):
  M, N = magnitude.shape
  Z = np.zeros((M, N), dtype=np.float32)
  angle = direction * 180. / np.pi
  angle[angle < 0] += 180
  for i in range(1, M-1):
    for j in range(1, N-1):
      q = 255
      r = 255
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

# Double threshold
def threshold (img, lowRatio=0.05, highRatio=0.15):
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

# Hysteresis
def hysteresis(img, weak, strong=255):
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


def canny_edge_detection(image_path):
  img = mpimg.imread(image_path)
  gray = rgb2gray(img)
  blurred = gaussian_blur(gray, 5, 1)
  mag, direction = sobel_filters(blurred)
  suppressed = non_maximum_suppression(mag, direction)
  thresholded, weak, strong = threshold(suppressed)
  final_img = hysteresis(thresholded, weak, strong)
  return final_img

for key in uploaded.keys():
  filename = key
  break

edges = canny_edge_detection(filename)
original = mpimg.imread(filename)

plt.figure(figsize=(12,5))

#input image
plt.subplot(1,2,1)
plt.imshow(original)
plt.title("Input Image")
plt.axis('off')

#output image
plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection (Manual)")
plt.axis('off')

plt.tight_layout()
plt.show()
