import cv2
import numpy as np
from matplotlib import pyplot as plt

# Open the images
image1 = cv2.imread('pochacco.jpg')
image2 = cv2.imread('pompompurin.jpg')

# Show the original images
cv2.imshow('Image 1', image1)
cv2.imshow('Image 2', image2)
cv2.waitKey(0)

# Calculate the histograms of the images
hist1 = cv2.calcHist([image1], [0], None, [256], [0,256])
hist2 = cv2.calcHist([image2], [0], None, [256], [0,256])

# Plot the histograms of the images
plt.figure(figsize=(12, 6))
plt.plot(hist1, color='blue', label='Image 1')
plt.plot(hist2, color='red', label='Image 2')
plt.title('Histograms of 2 Different Images')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Separate RGB histograms for each image
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist1 = cv2.calcHist([image1], [i], None, [256], [0,256])
    hist2 = cv2.calcHist([image2], [i], None, [256], [0,256])
    plt.plot(hist1, color=col, label=f'Image 1 {col.upper()}')
    plt.plot(hist2, color=col, linestyle='dashed', label=f'Image 2 {col.upper()}')
plt.title('Separate RGB Histograms')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

cv2.destroyAllWindows()
