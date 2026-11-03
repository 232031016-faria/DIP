import os

import cv2
import matplotlib.pyplot as plt

# Use paths relative to this script so it works from any working directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "Output")
os.makedirs(output_dir, exist_ok=True)

# Read image in grayscale
input_path = os.path.join(script_dir, "input.jpg")
img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Input image not found: {input_path}")

# Apply histogram equalization
equalized = cv2.equalizeHist(img)

# Display images
plt.figure(figsize=(12, 7))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(equalized, cmap='gray')
plt.title("Equalized Image")
plt.axis("off")

# Original histogram
plt.subplot(2, 2, 3)
plt.hist(img.ravel(), 256, [0, 256])
plt.title("Original Histogram")

# Equalized histogram
plt.subplot(2, 2, 4)
plt.hist(equalized.ravel(), 256, [0, 256])
plt.title("Equalized Histogram")

plt.tight_layout()

# Save both the equalized image and the visualization in the Output folder.
equalized_path = os.path.join(output_dir, "equalized_image.jpg")
plot_path = os.path.join(output_dir, "histogram_equalization.jpg")

if not cv2.imwrite(equalized_path, equalized):
    raise OSError(f"Could not save equalized image: {equalized_path}")

plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Saved equalized image to: {equalized_path}")
print(f"Saved histogram visualization to: {plot_path}")

plt.show()
plt.close()