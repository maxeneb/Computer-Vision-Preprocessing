import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Specify the image filename
filename = 'pochacco.jpg'

# Open the image
image = cv2.imread(filename)

# Get the properties of the image
print(f'Filename: {filename}')
print(f'Format: RGB')
print(f'Width: {image.shape[1]} pixels')
print(f'Height: {image.shape[0]} pixels')
print(f'Size: {os.path.getsize(filename)} bytes')

# Get the value of a pixel in the image (R,G,B)
x, y = 50, 50  # replace with your coordinates
pixel_value = image[y, x]
print(f'The RGB value at ({x}, {y}) is {pixel_value}')

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

plt.subplot(222), plt.imshow(edges, cmap='gray')
plt.title('Edges of Image'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.plot(hist)
plt.title('Histogram of Grayscale Image'), plt.xlabel('Pixel Value'), plt.ylabel('Frequency')

plt.show()
