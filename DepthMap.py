# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:01:18 2019

@author: USUARIO
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('im0.png', 0)  # 0 flag returns a grayscale image
imgR = cv2.imread('im1.png', 0)
imgL2 = cv2.imread('im0U.png', 0)  # 0 flag returns a grayscale image
imgR2 = cv2.imread('im1U.png', 0)
imgL3 = cv2.imread('im0M.png', 0)  # 0 flag returns a grayscale image
imgR3 = cv2.imread('im1M.png', 0)


block = 11
chanel =3 

# emperical values from P1, P2 as suggested in Ocv documentation
P1 = block * block * 8*chanel   # block * block * 8
P2 = block * block * 32*chanel # block * block * 32

stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=8*16, blockSize=block, P1=P1, P2=P2)


disparity = stereo.compute(imgL,imgR)
disparity2 = stereo.compute(imgL2,imgR2)
disparity3 = stereo.compute(imgL3,imgR3)
plt.imshow(disparity,'gray')
plt.show()
plt.imshow(disparity2,'gray')
plt.show()
plt.imshow(disparity3,'gray')
plt.show()

cv2.imwrite('bin.png', disparity)


ImT1_L = cv2.imread('Leti/indoor_forward_11_snapdragon/img/image_0_1020.png', 0)  # 0 flag returns a grayscale image
ImT1_R = cv2.imread('Leti/indoor_forward_11_snapdragon/img/image_1_1020.png', 0)


block = 5
chanel =3 

# emperical values from P1, P2 as suggested in Ocv documentation
P1 = block * block * 8*chanel   # block * block * 8
P2 = block * block * 32*chanel # block * block * 32

stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=8*16, blockSize=block, P1=P1, P2=P2)


disparity = stereo.compute(imgL,imgR)
plt.show()

cv2.imwrite('d.png', disparity)
"""
featureEngine = cv2.FastFeatureDetector_create()
TILE_H = 10
TILE_W = 20
H,W = imgL.shape    
kp = []
kpR = []
idx = 0
for y in range(0, H, TILE_H):
    for x in range(0, W, TILE_W):
        imPatch = imgL[y:y+TILE_H, x:x+TILE_W]
        keypoints = featureEngine.detect(imPatch)
        imPatchR = imgR[y:y+TILE_H, x:x+TILE_W]
        keypointsR = featureEngine.detect(imPatchR)
        for pt in keypoints:
            pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

        if (len(keypoints) > 10):
            keypoints = sorted(keypoints, key=lambda x: -x.response)
            for kpt in keypoints[0:10]:
                kp.append(kpt)
        else:
            for kpt in keypoints:
                kp.append(kpt)
        for ptR in keypointsR:
            ptR.pt = (ptR.pt[0] + x, ptR.pt[1] + y)

        if (len(keypointsR) > 10):
            keypointsR = sorted(keypointsR, key=lambda x: -x.response)
            for kptR in keypointsR[0:10]:
                kpR.append(kptR)
        else:
            for kptR in keypoints:
                kpR.append(kptR)

ftDebug = imgL
ftDebug = cv2.drawKeypoints(imgL, kp, ftDebug, color=(255,0,0))
ftDebugR = imgR
ftDebugR = cv2.drawKeypoints(imgR, kpR, ftDebugR, color=(255,0,0))

cv2.imshow('left', ftDebug)
cv2.imshow('right', ftDebugR)
cv2.waitKey()
cv2.destroyAllWindows()


trackPoints1 = cv2.KeyPoint_convert(kp)
trackPoints1 = np.expand_dims(trackPoints1, axis=1)
lk_params = dict( winSize  = (15,15),
                          maxLevel = 3,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

trackPoints2, st, err = cv2.calcOpticalFlowPyrLK(imgR,imgL, trackPoints1, None, flags=cv2.MOTION_AFFINE, **lk_params)

ptTrackable = np.where(st == 1, 1,0).astype(bool)
trackPoints1_KLT = trackPoints1[ptTrackable, ...]
trackPoints2_KLT_t = trackPoints2[ptTrackable, ...]
trackPoints2_KLT = np.around(trackPoints2_KLT_t)

error = 4
errTrackablePoints = err[ptTrackable, ...]
errThresholdedPoints = np.where(errTrackablePoints < error, 1, 0).astype(bool)
trackPoints1_KLT = trackPoints1_KLT[errThresholdedPoints, ...]
trackPoints2_KLT = trackPoints2_KLT[errThresholdedPoints, ...]
"""
"""
trackPointsL = np.zeros((len(kp), 1, 2), dtype=np.float32)
trackPointsR = np.zeros((len(kpR), 1, 2), dtype=np.float32)
for i, kpt in enumerate(kp):
    trackPoints1[i, :, 0] = kpt.pt[0]
    trackPoints1[i, :, 1] = kpt.pt[1]
"""
