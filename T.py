# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:16:14 2019

@author: USUARIO
"""

import numpy as np
import cv2

ImT1_L = cv2.imread('Leti/indoor_forward_11_snapdragon/img/image_0_1020.png', 0)  # 0 flag returns a grayscale image
ImT1_R = cv2.imread('Leti/indoor_forward_11_snapdragon/img/image_1_1020.png', 0)

ImT2_L = cv2.imread('Leti/indoor_forward_11_snapdragon/img/image_0_1021.png', 0)
ImT2_R = cv2.imread('Leti/indoor_forward_11_snapdragon/img/image_1_1021.png', 0)

# cv2.imshow('ImT1_L', ImT1_L)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

block = 5

# emperical values from P1, P2 as suggested in Ocv documentation
P1 = block * block * 8   # block * block * 8 * chanel
P2 = block * block * 32 # block * block * 32 * chanel

disparityEngine = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=block, P1=P1, P2=P2)
ImT1_disparity = disparityEngine.compute(ImT1_L, ImT1_R).astype(np.float32)
cv2.imwrite('disparity.png', ImT1_disparity)
ImT1_disparityA = np.divide(ImT1_disparity, 16.0)

ImT2_disparity = disparityEngine.compute(ImT2_L, ImT2_R).astype(np.float32)
ImT2_disparityA = np.divide(ImT2_disparity, 16.0)

fastFeatureEngine = cv2.FastFeatureDetector_create()

keypoints = fastFeatureEngine.detect(ImT1_L)
ftDebug = ImT1_L
ftDebug = cv2.drawKeypoints(ImT1_L, keypoints, ftDebug, color=(255,0,0))
plt.imshow(ftDebug,'gray')
plt.show()
#cv2.imwrite('ftDebug.png', ftDebug)

"""
TILE_H = 100
TILE_W = 100
H, W = ImT1_L.shape
kp = []
idx = 0
for y in range(0, H, TILE_H):
    for x in range(0, W, TILE_W):
        imPatch = ImT1_L[y:y + TILE_H, x:x + TILE_W]
        keypoints = fastFeatureEngine.detect(imPatch)
        for pt in keypoints:
            pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

        if (len(keypoints) > 10):
            keypoints = sorted(keypoints, key=lambda x: -x.response)
            for kpt in keypoints[0:10]:
                kp.append(kpt)
        else:
            for kpt in keypoints:
                kp.append(kpt)

kp = keypoints

ftDebug = ImT1_L
ftDebug = cv2.drawKeypoints(ImT1_L, kp, ftDebug, color=(255, 0, 0))
plt.imshow(ftDebug,'gray')
plt.show()
"""