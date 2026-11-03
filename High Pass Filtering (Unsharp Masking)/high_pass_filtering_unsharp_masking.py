import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread("High Pass Filtering (Unsharp Masking)/input.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: input.jpg not found!")
    exit()

# Apply Gaussian blur
blur = cv2.GaussianBlur(img, (5, 5), 1.0)

# Unsharp masking
mask = cv2.subtract(img, blur)

# Add mask to original image
sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

# Display results
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title("High Pass Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sharp, cmap='gray')
plt.title("Unsharp Masking")
plt.axis("off")

plt.tight_layout()
plt.savefig("unsharp_masking.jpg")
plt.show()