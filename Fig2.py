# H. Zeng, MDM Morteza, M. Goljan, 2023
# Packages
import numpy as np
from matplotlib import pyplot as plt
from skimage import io

from utils_DWT.NoiseExtract import NoiseExtract
from utils_DTCWT.NoiseExtract_DTCWT import NoiseExtract_DTCWT

# 1 读入图片并裁剪
clean = io.imread('images/circlesBrightDark.png')
clean = clean[350:350 + 128, 200:200 + 128].astype(np.float32)
# 2 加噪
sigma = 20  # 5, 20
Fingerprint = sigma * np.random.randn(128, 128)
noisy = clean + Fingerprint
noisy = np.clip(noisy, 0, 255)  # Python 和 Matlab 的uint8()不同

plt.subplot(221)
plt.imshow(np.uint8(clean), cmap='gray')
plt.subplot(222)
plt.imshow(np.uint8(noisy), cmap='gray')

# 3 DWT 去噪
L = 4
qmf = [.230377813309, .714846570553, .630880767930, -0.027983769417,
       -0.187034811719, .030841381836, .032883011667, -0.010597401785]
noise = NoiseExtract(noisy, qmf, sigma, L)
denoised_dwt = np.clip(noisy - noise, 0, 255)
plt.subplot(223)
plt.imshow(np.uint8(denoised_dwt), cmap='gray')

# 4 DTCWT 去噪
# denoised_dtcwt = denC2D_dwt(noisy,sigma);
noise_dtcwt = NoiseExtract_DTCWT(noisy, sigma, L)
denoised_dtcwt = np.clip(noisy - noise_dtcwt, 0, 255)
plt.subplot(224)
plt.imshow(np.uint8(denoised_dtcwt), cmap='gray')
plt.show()
Done = 1
