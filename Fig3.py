# H. Zeng, MDM Morteza, M. Goljan, 2023
# Current DTCWT package (python) does not support customized extension mode as Matlab.
# Packages
import numpy as np
from matplotlib import pyplot as plt
from skimage import io

from utils_DWT.NoiseExtract import NoiseExtract
from utils_DTCWT.NoiseExtract_DTCWT import NoiseExtract_DTCWT

# 1 读入图片并裁剪
clean = io.imread('images/circlesBrightDark.png')
clean = clean[350:350 + 128, 200:200 + 128].astype(np.float32)
# 2 Parameters
iter = 200
sigma = 5  # 5, 20
L = 4
qmf = [.230377813309, .714846570553, .630880767930, -0.027983769417,
       -0.187034811719, .030841381836, .032883011667, -0.010597401785]
Similarity_dwt = np.zeros([128, 128], dtype=np.float32)
Similarity_dtcwt = np.zeros([128, 128], dtype=np.float32)

# 3 Calculate the S-map, Dark area indicates low PRNU quality
for i in range(iter):
    if i % 10 == 0:
        print(i)
    Fingerprint = sigma * np.random.randn(128, 128)
    noisy = clean + Fingerprint
    noisy = np.clip(noisy, 0, 255)  # Python 和 Matlab 的uint8()不同

    noise = NoiseExtract(noisy, qmf, sigma, L)
    noise_dtcwt = NoiseExtract_DTCWT(noisy, sigma, L)

    Similarity_dwt = Similarity_dwt + noise * Fingerprint
    Similarity_dtcwt = Similarity_dtcwt + noise_dtcwt * Fingerprint

plt.subplot(121)
plt.imshow(Similarity_dwt/iter, cmap='gray', vmin=0, vmax=10)
plt.title('DWT')

plt.subplot(122)
plt.imshow(Similarity_dtcwt/iter, cmap='gray', vmin=0, vmax=10)
plt.title('DTCWT')
plt.show()
print('Dark area indicates low PRNU quality')
Done = 1


