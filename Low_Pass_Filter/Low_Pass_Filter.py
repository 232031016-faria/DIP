import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

out_dir = 'DIP_Frequency_Assignments_Results'
os.makedirs(out_dir, exist_ok=True)
root = Tk(); root.withdraw(); root.attributes('-topmost', True)
path = filedialog.askopenfilename(title='Select an image', filetypes=[('Images','*.jpg *.jpeg *.png *.bmp')])
root.destroy()
if not path: raise SystemExit('No image selected.')
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
if image is None: raise SystemExit('Could not read image.')
fft_shift = np.fft.fftshift(np.fft.fft2(image))
rows, cols = image.shape
Y, X = np.ogrid[:rows, :cols]
center_y, center_x = rows//2, cols//2
radius = 40
distance = np.sqrt((X-center_x)**2 + (Y-center_y)**2)
mask = (distance <= radius).astype(np.float32)
filtered_fft = fft_shift * mask
filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))
filtered = np.clip(filtered, 0, 255)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(image, cmap='gray'); plt.title('Original Image'); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(filtered, cmap='gray'); plt.title('Low-Pass Filtered Image'); plt.axis('off')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'Assignment_4_Low_Pass_Filter.png'), dpi=300); plt.show(); plt.close()
print('Assignment 4 completed. Output saved in', out_dir)
