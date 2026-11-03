import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread("Spatial Filtering (Correlation)/input.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: input.jpg not found!")
    exit()

# Define 3x3 averaging mask
kernel = np.ones((3, 3), dtype=np.float32) / 9

# Apply correlation using filter2D
output = cv2.filter2D(img, -1, kernel)

# Display results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title("Spatial Filtering - Correlation")
plt.axis("off")

plt.tight_layout()
plt.savefig("spatial_filtering_correlation.jpg")
plt.show()