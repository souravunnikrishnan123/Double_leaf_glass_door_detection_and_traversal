import cv2
import numpy as np

def get_rgb_based_lines_using_canny_and_hough_lines(color_image):
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(16, 16))
    contrast = clahe.apply(gray)

    #filtered = cv2.bilateralFilter(contrast, d=9, sigmaColor=150, sigmaSpace=100)
    filtered = cv2.GaussianBlur(contrast, (5, 5), 1.0)

    median_val = np.median(filtered)
    lower = int(max(0, 0.7 * median_val))
    upper = int(min(255, 2.0 * median_val))

    edges = cv2.Canny(filtered, lower, upper)

    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #edges = cv2.Canny(blurred, 50, 150)  # Canny edge detector
    
    # Hough Line Transform to detect lines
    # Detect lines using Probabilistic Hough Transform
    # Parameters:
    #  - 1: pixel resolution of the Hough grid
    #  - np.pi / 180: angle resolution in radians (1 degree)
    #  - threshold=100: minimum number of intersections to detect a line
    #  - minLineLength=100: minimum length of line in pixels to be considered
    #  - maxLineGap=10: maximum allowed gap between line segments to link them
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=20)  #maxlingap of 30px is needed to detect door handle
    # minLineLength of 100px is needed to detect brown doors where the frame is segmented horizontally
    # Draw detected vertical lines on image
    return lines, edges
