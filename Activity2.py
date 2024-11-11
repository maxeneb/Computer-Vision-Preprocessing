import cv2
import numpy as np

# Open image
image = cv2.imread('pompompurin.jpg')

# Get the size of the image
height, width, channels = image.shape
print(f'The image size is {width}x{height} and it has {channels} channels.')

# Get the value of a pixel in the image (R,G,B)
x, y = 50, 50  # replace with your coordinates
b, g, r = image[y, x]
print(f"Pixel at ({x}, {y}): BGR = ({b}, {g}, {r})")

# Show the 4 different edges of an image
# edges = cv2.Canny(image,100,200) # parameters can be changed based on the image
# cv2.imshow('Edges', edges)
# cv2.waitKey(0)

# Convert to grayscale
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

# Canny Edge Detection
edges_canny = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
cv2.imshow('Canny Edge Detection', edges_canny)
cv2.waitKey(0)

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

# Display Sobel Edge Detection Images
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y', sobelxy)
cv2.waitKey(0)


# Show the RGB colors of the same image separately
blue, green, red = cv2.split(image)

cv2.imshow('Red channel', red)
cv2.waitKey(0)

cv2.imshow('Green channel', green)
cv2.waitKey(0)

cv2.imshow('Blue channel', blue)
cv2.waitKey(0)

cv2.destroyAllWindows()
