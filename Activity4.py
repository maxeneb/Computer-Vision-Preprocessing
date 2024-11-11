import cv2
import numpy as np
from matplotlib import pyplot as plt

# Open the image
image = cv2.imread('pochacco.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate the histogram of the grayscale image
hist = cv2.calcHist([gray_image], [0], None, [256], [0,256])

# Detect the edges of the image
edges = cv2.Canny(gray_image, 100, 200)

# Display all images in one figure
plt.figure(figsize=(10, 10))

plt.subplot(221), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.plot(hist)
plt.title('Histogram of Grayscale Image'), plt.xlabel('Pixel Value'), plt.ylabel('Frequency')

plt.subplot(224), plt.imshow(edges, cmap='gray')
plt.title('Edges of Image'), plt.xticks([]), plt.yticks([])

plt.show()
