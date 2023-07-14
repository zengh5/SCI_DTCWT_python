# dtcwt packeage is needed, 'https://dtcwt.readthedocs.io/en/0.12.0/index.html'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

import dtcwt


def NoiseExtract_DTCWT(img, sigma, L=4):
    NoiseVar = np.power(sigma, 2)
    # Note for DTCWT, two wavelets should be specified, one for the first level, the other for level>=2
    # you are free to try other wavelets, e.g., 'legall', 'near_sym_b'
    dtcwt2D = dtcwt.Transform2d(biort='near_sym_a', qshift='qshift_06')
    img_dtcwt_coe = dtcwt2D.forward(img, nlevels=L)
    Low_band = img_dtcwt_coe.lowpass
    High_band = img_dtcwt_coe.highpasses
    High_band_noise = np.copy(High_band)      # noise component of High frequency bands

    for i in range(0, L):
        B0 = High_band[i][:, :, 0]
        B1 = High_band[i][:, :, 1]
        B2 = High_band[i][:, :, 2]
        B3 = High_band[i][:, :, 3]
        B4 = High_band[i][:, :, 4]
        B5 = High_band[i][:, :, 5]
        # 此处要理解cwt的结构，对复小波系数进行去噪，这是整个算法的灵魂所在
        noise_B0 = WaveNoise_abs(B0, NoiseVar)
        noise_B1 = WaveNoise_abs(B1, NoiseVar)
        noise_B2 = WaveNoise_abs(B2, NoiseVar)
        noise_B3 = WaveNoise_abs(B3, NoiseVar)
        noise_B4 = WaveNoise_abs(B4, NoiseVar)
        noise_B5 = WaveNoise_abs(B5, NoiseVar)

        High_band_noise[i][:, :, 0] = noise_B0
        High_band_noise[i][:, :, 1] = noise_B1
        High_band_noise[i][:, :, 2] = noise_B2
        High_band_noise[i][:, :, 3] = noise_B3
        High_band_noise[i][:, :, 4] = noise_B4
        High_band_noise[i][:, :, 5] = noise_B5

    img_dtcwt_coe.highpasses = High_band_noise
    # We only interest the noise component, thus set the low-frequency component to zero,
    img_dtcwt_coe.lowpass[:, :] = 0
    noise_reconstruct = dtcwt2D.inverse(img_dtcwt_coe)

    return noise_reconstruct


# 算法思想：相邻小波系数之间具有相关性，具体来说就是小波系数的局部方差是平稳的：一块变化剧烈的小波系数周围
# 区域的小波系数变化往往也会剧烈，一块平坦的小波系数周围区域的小波系数往往也会平坦
def WaveNoise_abs(coef, NoiseVar):
    # 因为小波系数每个区域的均值约为0，D(X) = E(X^2)-(E(X)^2) =E(X^2)
    # 所以EstVar1就是局部方差
    tc = abs(coef)*abs(coef)    # 此处有区别！
    # cv2.blur() 是求邻域均值
    EstVar1 = cv2.blur(tc, (3, 3))
    # 把方差小于NoiseVar的值置为0，这就是去噪
    temp = EstVar1 - NoiseVar
    coefVar = np.maximum(temp,0)
    # 为了算法更可靠，我们用不同大小的窗口来计算方差，取其中的最小值为最终的方差
    for w in range(5, 10, 2):
        EstVar1 = cv2.blur(tc, (w, w))
        temp = EstVar1 - NoiseVar
        EstVar = np.maximum(temp, 0)
        coefVar = np.minimum(coefVar, EstVar)

    # 理解这个公式，coefVar=0 时，说明这个位置小波系数全部是噪声，所以tc =coef
    #            coefVar>>NoiseVar时，说明这个位置小波系数主要是图像纹理，所以tc = 0
    tc = (coef * NoiseVar)/(coefVar + NoiseVar)
    return tc
