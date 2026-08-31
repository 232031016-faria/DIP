import cv2
import numpy as np
import os

# Correct path to input image
img = cv2.imread("linear_transformation/input.jpg")

# Check if image was loaded
if img is None:
    print("Error: input.jpg not found!")
    exit()

# Linear transformation
alpha = 1.5
beta = 30

output = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# Display images
cv2.imshow("Original Image", img)
cv2.imshow("Linear Transformation", output)

# Save output
cv2.imwrite("linear_transformation/linear_transformation.jpg", output)

cv2.waitKey(0)
cv2.destroyAllWindows()