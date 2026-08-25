import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
import os
import cv2

# Get the folder where this Python file is located
folder = os.path.dirname(os.path.abspath(__file__))

# Create the complete path to input.jpg
image_path = os.path.join(folder, "input.jpg")

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Error: input.jpg not found!")
    exit()

print("Image loaded successfully!")

# Convert BGR to RGB
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Normalize RGB values to 0-1
rgb_float = rgb.astype(np.float32) / 255.0

r = rgb_float[:, :, 0]
g = rgb_float[:, :, 1]
b = rgb_float[:, :, 2]

# RGB to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -----------------------------
# RGB to HSI
# -----------------------------

# Intensity
I = (r + g + b) / 3.0

# Saturation
minimum = np.minimum(np.minimum(r, g), b)
S = np.zeros_like(I)
non_zero = I > 0
S[non_zero] = 1 - (minimum[non_zero] / I[non_zero])

# Hue
numerator = 0.5 * ((r - g) + (r - b))
denominator = np.sqrt((r - g)**2 + (r - b) * (g - b))
denominator = np.where(denominator == 0, 1e-10, denominator)

theta = np.arccos(np.clip(numerator / denominator, -1, 1))
H = np.where(b <= g, theta, 2 * np.pi - theta)

# Convert Hue from radians to degrees
H = np.degrees(H)

# Normalize Hue for display
H_display = H / 360.0

# -----------------------------
# Display results
# -----------------------------

plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.imshow(rgb)
plt.title("Original RGB Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(H_display, cmap="hsv")
plt.title("HSI - Hue")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(S, cmap="gray")
plt.title("HSI - Saturation")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(I, cmap="gray")
plt.title("HSI - Intensity")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(H_display, cmap="hsv")
plt.title("HSI Hue Visualization")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save components
cv2.imwrite("grayscale.jpg", gray)
cv2.imwrite("hue.jpg", (H_display * 255).astype(np.uint8))
cv2.imwrite("saturation.jpg", (S * 255).astype(np.uint8))
cv2.imwrite("intensity.jpg", (I * 255).astype(np.uint8))

print("RGB to Grayscale and HSI conversion completed successfully.")
