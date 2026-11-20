import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

out_dir = 'DIP_Frequency_Assignments_Results'
os.makedirs(out_dir, exist_ok=True)
def select_image(title):
    root = Tk(); root.withdraw(); root.attributes('-topmost', True)
    path = filedialog.askopenfilename(title=title, filetypes=[('Images','*.jpg *.jpeg *.png *.bmp')])
    root.destroy(); return path
p1 = select_image('Select Image 1')
p2 = select_image('Select Image 2')
if not p1 or not p2: raise SystemExit('Both images must be selected.')
img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
if img1 is None or img2 is None: raise SystemExit('Could not read one of the images.')
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
f1 = np.fft.fftshift(np.fft.fft2(img1)); f2 = np.fft.fftshift(np.fft.fft2(img2))
mag1, phase1 = np.abs(f1), np.angle(f1)
mag2, phase2 = np.abs(f2), np.angle(f2)
# Image 1 magnitude with Image 2 phase
combined1 = mag1 * np.exp(1j * phase2)
# Image 2 magnitude with Image 1 phase
combined2 = mag2 * np.exp(1j * phase1)
def inverse_display(freq):
    spatial = np.fft.ifft2(np.fft.ifftshift(freq))
    return np.clip(np.abs(spatial), 0, 255)
result1, result2 = inverse_display(combined1), inverse_display(combined2)
plt.figure(figsize=(12,5))
for i,(im,title) in enumerate([(img1,'Image 1'),(img2,'Image 2'),(result1,'Magnitude 1 + Phase 2'),(result2,'Magnitude 2 + Phase 1')],1):
    plt.subplot(1,4,i); plt.imshow(im, cmap='gray'); plt.title(title); plt.axis('off')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'Assignment_3_Magnitude_Phase_Exchange.png'), dpi=300); plt.show(); plt.close()
print('Assignment 3 completed. Output saved in', out_dir)
