import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import imghdr

def display_menu():
    print("\n=== GVC ACTIVITIES ===")
    print("1. Open Activity 1")
    print("2. Open Activity 2")
    print("3. Open Activity 3")
    print("4. Open Activity 4")
    print("5. Open Activity 5")
    print("6. Exit")

def activity1(image_path):
    # 1. open image
    img1 = cv2.imread(image_path)
    print("shape of the load", img1.shape)

    # 2. display image using the different flags

        # cv2.IMREAD_COLOR, value = 1
    color_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imshow('Flag 1: Color Image', color_img)
    cv2.waitKey(0)

        # cv2.IMREAD_GRAYSCALE, value = 0
    grayscale_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Flag 2: Grayscale Image', grayscale_img)
    cv2.waitKey(0)

        # cv2.IMREAD_UNCHANGED, value = -1
    unchanged_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    cv2.imshow('Flag 3: Unchanged Image', unchanged_img)
    cv2.waitKey(0)

    # 3. convert image to grayscale
    gray_image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.waitKey(0)

    # 4. get the image value of a pixel
    x, y = 50, 50  
    pixel_value = gray_image[y, x]
    print(f'The pixel value at ({x}, {y}) is {pixel_value}')

    # 5. convert the image to black and white
    _, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Black and White Image', bw_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def activity2(image_path):
    # read image
    image = cv2.imread(image_path)

    # 1. get the size of the image
    height, width, channels = image.shape
    print(f'The image size is {width}x{height} and it has {channels} channels.')

    # 2. get the value of a pixel in the image (R,G,B)
    x, y = 50, 50 
    b, g, r = image[y, x]
    print(f"Pixel at ({x}, {y}): BGR = ({b}, {g}, {r})")

    # 3. show the 4 different edges of an image (using diff. paramets/method)
        # convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # apply gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
        # canny edge detection
    edges_canny = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    cv2.imshow('Canny Edge Detection', edges_canny)
    cv2.waitKey(0)

        # sobel edge detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        # display sobel edge detection images
    cv2.imshow('Sobel X', sobelx)
    cv2.waitKey(0)
    cv2.imshow('Sobel Y', sobely)
    cv2.waitKey(0)
    cv2.imshow('Sobel X Y', sobelxy)
    cv2.waitKey(0)


    # 4. show the RGB colors of the same image separately
    blue, green, red = cv2.split(image)

        #red
    cv2.imshow('Red channel', red)
    cv2.waitKey(0)

        #green
    cv2.imshow('Green channel', green)
    cv2.waitKey(0)

        #blue
    cv2.imshow('Blue channel', blue)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def activity3(image_path_1, image_path_2):
    # read the images
    image1 = cv2.imread(image_path_1)
    image2 = cv2.imread(image_path_2)

    # 1. show the original images
    cv2.imshow('Image 1', image1)
    cv2.imshow('Image 2', image2)
    cv2.waitKey(0)

    #2. show the histogram of 2 different images

        #calculating the histograms
    hist1 = cv2.calcHist([image1], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0,256])

        # plotting the histograms
    plt.figure(figsize=(12, 6))
    plt.plot(hist1, color='blue', label='Image 1')
    plt.plot(hist2, color='red', label='Image 2')
    plt.title('Histograms of 2 Different Images')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # 3. separate RGB histograms
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

def activity4(image_path):
    # 1. display original image (colored)
    image = cv2.imread('pochacco.jpg')

    # 2. convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. show the histogram of the grayscale image
    hist = cv2.calcHist([gray_image], [0], None, [256], [0,256])

    # 4. display the edges of the image
    edges = cv2.Canny(gray_image, 100, 200)

    # 5. display/plot all images in one figure
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

def activity5(image_path):
    #read image
    image = cv2.imread(image_path)

    #1. make a program that wil display/plot all images in one figure only

        # convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # calculate the histogram of the grayscale image
    hist = cv2.calcHist([gray_image], [0], None, [256], [0,256])

        # detect the edges of the image
    edges = cv2.Canny(gray_image, 100, 200)

        # display all images in one figure
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

    # 2. print the properties of the image
    print("- PROPERTIES OF THE IMAGE -")
    print(f'Filename: {image_path}')
    print(f'Format: {imghdr.what(image_path).upper()}')
    print(f'Width: {image.shape[1]} pixels')
    print(f'Height: {image.shape[0]} pixels')
    print(f'Size: {os.path.getsize(image_path)} bytes')

    # 3. get the value of a pixel in the image
    x, y = 50, 50 
    pixel_value = image[y, x]
    print(f'The RGB value at ({x}, {y}) is {pixel_value}')

def main():
    while True:
        display_menu()
        choice = input("Enter your choice: ")
        if choice == "1":
            activity1('pochacco.jpg')
        elif choice == "2":
            activity2('pompompurin.jpg')
        elif choice == "3":
            activity3('pochacco.jpg', 'pompompurin.jpg')
        elif choice == "4":
            activity4('pochacco.jpg')
        elif choice == "5":
            activity5('pompompurin.jpg')
        elif choice == "6":
            print("Thankyou for using~!")
            break
        else:
            print("Invalid choice. Please choose again.")

if __name__ == "__main__":
    main()
