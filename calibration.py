import cv2
import numpy as np
import matplotlib.image as mpimg


class CameraPreprocessing:
    def __init__(self, path):
        self.path = path

    def calibrate_camera(self):
        nx = 9
        ny = 6

        objpoints = []
        imgpoints = []

        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        for each_img_path in self.path:
            # load image
            img = mpimg.imread('camera_cal/' + each_img_path)
            # gray scale the image
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # find corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)
                # draw corners on the image
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist

    def undistort(self, img):
        mtx, dist = self.calibrate_camera()
        undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
        return undistorted_img
