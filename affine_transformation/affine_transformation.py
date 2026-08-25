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
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width = image.shape[:2]

# 1. Translation
tx = 100
ty = 50

translation_matrix = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

translated = cv2.warpAffine(
    image, translation_matrix, (width, height)
)

# 2. Rotation
center = (width // 2, height // 2)

rotation_matrix = cv2.getRotationMatrix2D(
    center, 45, 1.0
)

rotated = cv2.warpAffine(
    image, rotation_matrix, (width, height)
)

# 3. Scaling
scaled = cv2.resize(
    image, None, fx=0.7, fy=0.7
)

# 4. Shearing
shear_matrix = np.float32([
    [1, 0.3, 0],
    [0.2, 1, 0]
])

sheared = cv2.warpAffine(
    image, shear_matrix, (width, height)
)

# Convert results to RGB
translated_rgb = cv2.cvtColor(translated, cv2.COLOR_BGR2RGB)
rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
scaled_rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
sheared_rgb = cv2.cvtColor(sheared, cv2.COLOR_BGR2RGB)

# Display
plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(translated_rgb)
plt.title("Translation")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(rotated_rgb)
plt.title("Rotation")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(scaled_rgb)
plt.title("Scaling")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(sheared_rgb)
plt.title("Shearing")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save transformed images to the same folder as this script
cv2.imwrite(os.path.join(folder, "translated_rgb.jpg"), cv2.cvtColor(translated_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(folder, "rotated_rgb.jpg"), cv2.cvtColor(rotated_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(folder, "scaled_rgb.jpg"), cv2.cvtColor(scaled_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(folder, "sheared_rgb.jpg"), cv2.cvtColor(sheared_rgb, cv2.COLOR_RGB2BGR))

print("Affine transformations completed successfully.")
