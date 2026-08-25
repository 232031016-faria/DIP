import cv2
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
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get original dimensions
height, width = image.shape[:2]

# New dimensions: 2 times larger
new_width = width * 2
new_height = height * 2

# 1. Nearest Neighbor
nearest = cv2.resize(
    image,
    (new_width, new_height),
    interpolation=cv2.INTER_NEAREST
)

# 2. Bilinear
bilinear = cv2.resize(
    image,
    (new_width, new_height),
    interpolation=cv2.INTER_LINEAR
)

# 3. Bicubic
bicubic = cv2.resize(
    image,
    (new_width, new_height),
    interpolation=cv2.INTER_CUBIC
)

# Convert results to RGB
nearest_rgb = cv2.cvtColor(nearest, cv2.COLOR_BGR2RGB)
bilinear_rgb = cv2.cvtColor(bilinear, cv2.COLOR_BGR2RGB)
bicubic_rgb = cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB)

# Display results
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(nearest_rgb)
plt.title("Nearest Neighbor")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(bilinear_rgb)
plt.title("Bilinear Interpolation")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(bicubic_rgb)
plt.title("Bicubic Interpolation")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save results to the same folder as this script
cv2.imwrite(os.path.join(folder, "nearest_neighbor.jpg"), nearest)
cv2.imwrite(os.path.join(folder, "bilinear.jpg"), bilinear)
cv2.imwrite(os.path.join(folder, "bicubic.jpg"), bicubic)

print("Spatial interpolation completed successfully.")
