# H. Zeng, MDM Morteza, M. Goljan, 2023

import numpy as np
from matplotlib import pyplot as plt
# from imageio import imread
from skimage import io
from scipy.io import loadmat

from utils_DWT.NoiseExtract import NoiseExtract
from utils_DTCWT.NoiseExtract_DTCWT import NoiseExtract_DTCWT
from utils_common.crosscorr import crosscorr
from utils_common.PCE import PCE


L = 4
qmf = [.230377813309, .714846570553, .630880767930, -0.027983769417,
       -0.187034811719, .030841381836, .032883011667, -0.010597401785]
sigma_DTCWT = 1.8
sigma = 2
Threshold = 10     # patches whose PCE lower than this threshold are regarded as tampered

Fingerprint = loadmat('mat/FlatFingerprint.mat')['Fingerprint']
IxN = io.imread('images/demoPS.png')

M, N = IxN.shape
IxN = IxN.astype(np.float32)
KI = IxN * Fingerprint

PCE_block_DWT = np.zeros([int(np.floor(M / 128)), int(np.floor(N / 128))], dtype=np.float64)
PCE_block_DTCWT = np.zeros([int(np.floor(M / 128)), int(np.floor(N / 128))], dtype=np.float64)

for row in range(int(np.floor(M / 128))):
    for col in range(int(np.floor(N / 128))):
        Ixb = IxN[row * 128: row*128 + 128, col * 128: col * 128 + 128]
        KIb = KI[row * 128: row*128 + 128, col * 128: col * 128 + 128]
        # the proposed method
        noise_dtcwt = NoiseExtract_DTCWT(Ixb, sigma_DTCWT, L)
        C = crosscorr(noise_dtcwt, KIb)
        Out = PCE(C)
        PCE_block_DTCWT[row, col] = Out['PCE']

        # Baseline： DWT
        noise_dwt = NoiseExtract(Ixb, qmf, sigma, L)
        C = crosscorr(noise_dwt, KIb)
        Out = PCE(C)
        PCE_block_DWT[row, col] = Out['PCE']

plt.subplot(131)
plt.imshow(IxN, cmap='gray')
plt.title('Probe image')   # Fig. 9(b)
plt.subplot(132)
plt.imshow(PCE_block_DWT, cmap='gray', vmax=10, vmin=0)
plt.title('DWT')   # Fig. 9(c)
plt.subplot(133)
plt.imshow(PCE_block_DTCWT, cmap='gray', vmax=10, vmin=0)
plt.title('DTCWT')  # Fig. 9(d)
plt.show()
# Fewer false alarms in the resulted map with DTCWT
Done = 1

