import cv2

img = cv2.imread("image_thresholding/input.jpg", cv2.IMREAD_GRAYSCALE)

# Binary thresholding
threshold_value = 127

_, output = cv2.threshold(
    img,
    threshold_value,
    255,
    cv2.THRESH_BINARY
)

cv2.imshow("Original Image", img)
cv2.imshow("Thresholded Image", output)

cv2.imwrite("image_thresholding/thresholded_image.jpg", output)

cv2.waitKey(0)
cv2.destroyAllWindows()