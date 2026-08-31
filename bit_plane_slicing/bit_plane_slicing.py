import cv2
import numpy as np

img = cv2.imread("bit_plane_slicing/input.jpg", cv2.IMREAD_GRAYSCALE)

# Select bit plane
bit = 7

# Extract selected bit plane
output = np.bitwise_and(img, 1 << bit)

# Convert to binary image
output = np.where(output > 0, 255, 0).astype(np.uint8)

cv2.imshow("Original Image", img)
cv2.imshow(f"Bit Plane {bit}", output)

cv2.imwrite(f"bit_plane_slicing/bit_plane_{bit}.jpg", output)

cv2.waitKey(0)
cv2.destroyAllWindows()