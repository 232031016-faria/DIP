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
image = cv2.resize(image, (256,256))
fft_shift = np.fft.fftshift(np.fft.fft2(image))
magnitude = np.log1p(np.abs(fft_shift))
rows, cols = magnitude.shape
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
fig = plt.figure(figsize=(10,7)); ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, magnitude, cmap='viridis', linewidth=0, antialiased=True)
ax.set_title('3D Frequency Domain Magnitude'); ax.set_xlabel('Frequency X'); ax.set_ylabel('Frequency Y'); ax.set_zlabel('Magnitude')
plt.tight_layout(); plt.savefig(os.path.join(out_dir,'Assignment_2_3D_Frequency_Plot.png'), dpi=300); plt.show(); plt.close()
print('Assignment 2 completed. Output saved in', out_dir)
