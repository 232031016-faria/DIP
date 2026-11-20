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
fft = np.fft.fft2(image)
fft_shift = np.fft.fftshift(fft)
spectrum = np.log1p(np.abs(fft_shift))
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(image, cmap='gray'); plt.title('Spatial Domain'); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(spectrum, cmap='gray'); plt.title('Frequency Spectrum'); plt.axis('off')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'Assignment_1_Spatial_to_Frequency.png'), dpi=300); plt.show(); plt.close()
print('Assignment 1 completed. Output saved in', out_dir)
