import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("Output", exist_ok=True)

# Read image
img = cv2.imread("Normalised Histogram PDF/input.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: input.jpg not found!")
    exit()

# Calculate histogram
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# Calculate Normalised Histogram PDF
pdf = hist / (img.shape[0] * img.shape[1])

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.plot(pdf)
plt.title("Normalised Histogram PDF")
plt.xlabel("Pixel Intensity")
plt.ylabel("Probability")
plt.grid()

plt.tight_layout()

# Save result
plt.savefig("Output/normalised_histogram_pdf.png", dpi=300)

# Save PDF values
np.savetxt("Output/histogram_pdf_values.txt", pdf)

print("Results saved successfully in Output folder!")

plt.show()
plt.close()