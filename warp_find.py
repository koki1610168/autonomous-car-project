import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class WarpAndFindLane:
    def __init__(self, img):
        self.img = img

    def warp(self):
        src = np.float32([[280,  700], [600,  450],  [725,  450], [1125, 700]])
        dis = np.float32([[250,  720], [250,    0],  [1065,   0], [1065, 720]])

        M = cv2.getPerspectiveTransform(src, dis)
        Minv = cv2.getPerspectiveTransform(dis, src)
        warped = cv2.warpPerspective(
            self.img, M, (self.img.shape[1], self.img.shape[0]), flags=cv2.INTER_LINEAR)
        return warped, M, Minv

    def hist(self, warped_img):
        bottom_half = warped_img[warped_img.shape[0]//2:, :]

        histogram = np.sum(bottom_half, axis=0)

        return histogram
