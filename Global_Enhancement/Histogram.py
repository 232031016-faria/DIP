import cv2
import matplotlib.pyplot as plt

# Read the input image
image = cv2.imread("Global_Enhancement/input.jpg", cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded
if image is None:
    print("Error: Image not found!")
    exit()

# Apply Histogram Equalization
enhanced_image = cv2.equalizeHist(image)

# Display original and enhanced images
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title("Enhanced Image")
plt.axis("off")

# Original Histogram
plt.subplot(2, 2, 3)
plt.hist(image.ravel(), bins=256, range=[0, 256])
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")

# Enhanced Histogram
plt.subplot(2, 2, 4)
plt.hist(enhanced_image.ravel(), bins=256, range=[0, 256])
plt.title("Enhanced Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")

plt.tight_layout()
plt.show()

# Save the enhanced image
cv2.imwrite("Global_Enhancement/enhanced_image.jpg", enhanced_image)

print("Global histogram enhancement completed successfully!")