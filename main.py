#!/home/koki/miniconda3/bin/python

import cv2
import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from calibration import CameraPreprocessing
from color_and_gradient import ColorAndGradient
from warp_find import WarpAndFindLane


def process_img(img):
    chessboard_imgs = os.listdir('camera_cal')

    prepro = CameraPreprocessing(chessboard_imgs)
    undistorted_img = prepro.undistort(img)

    wp = WarpAndFindLane(undistorted_img)
    warped_image, M, Minv = wp.warp()
    graph = wp.hist(warped_image/255)

    cg = ColorAndGradient(warped_image)
    cg_combined_binary = cg.combined()

    #graph = wp.hist(warped_image)

    return undistorted_img, cg_combined_binary, warped_image


if __name__ == "__main__":

    cap = cv2.VideoCapture("project_video.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    image = mpimg.imread('test_images/test5.jpg')
    undist, comb, warp = process_img(image)

    f, (x1, x2, x3) = plt.subplots(1, 3, figsize=(24, 9))
    x1.axis('off')
    x1.imshow(undist)
    x1.set_title('Undistorted', fontsize=20)

    x2.axis('off')
    x2.imshow(warp)
    x2.set_title('Warp', fontsize=20)

    x3.axis('off')
    x3.imshow(comb, cmap='gray')
    x3.set_title('Color', fontsize=20)

    plt.show()
