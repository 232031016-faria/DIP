import cv2
import numpy as np

img = cv2.imread("gamma_correction/input.jpg")

gamma = 2.2

# Gamma correction
table = np.array([
    ((i / 255.0) ** (1 / gamma)) * 255
    for i in np.arange(256)
]).astype("uint8")

output = cv2.LUT(img, table)

cv2.imshow("Original Image", img)
cv2.imshow("Gamma Corrected Image", output)

cv2.imwrite("gamma_correction/gamma_correction.jpg", output)

cv2.waitKey(0)
cv2.destroyAllWindows()