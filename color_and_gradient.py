import matplotlib.image as mpimg
import cv2
import numpy as np


class ColorAndGradient:
    def __init__(self, img):
        self.img = img

    def abs_sobel_thresh(self, orient='x', thresh_min=0, thresh_max=255):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))

        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) &
                      (scaled_sobel <= thresh_max)] = 1

        return binary_output

    def dir_threshold(self, sobel_kernel=3, thresh=(0, np.pi/2)):
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        # calculate the direction of the gradient
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) &
                      (absgraddir <= thresh[1])] = 1

        return binary_output

    def mag_thresh(self, sobel_kernel=3, mag_thresh=(0, 255)):
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) &
                      (gradmag < mag_thresh[1])] = 1

        return binary_output

    def color(self):
        color_threshold = 200
        r_channel = self.img[:, :, 0]
        g_channel = self.img[:, :, 1]
        r_binary = np.zeros_like(r_channel)
        r_binary[(r_channel > color_threshold) &
                 (g_channel > color_threshold)] = 1

        hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]

        s_thresh_min = 50
        s_thresh_max = 150
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel < s_thresh_max)] = 1

        lab = cv2.cvtColor(self.img, cv2.COLOR_RGB2Lab)
        b_channel = lab[:, :, 2]

        b_thresh_min = 230
        b_thresh_max = 255
        b_binary = np.zeros_like(b_channel)
        b_binary[(b_channel >= b_thresh_min) & (b_channel < b_thresh_max)] = 1

        color_combined = np.zeros_like(s_channel)
        color_combined[(s_binary == 1) | (b_binary == 1)] = 1

        return color_combined

    def combined(self):
        gradx = self.abs_sobel_thresh(
            orient='x', thresh_min=20, thresh_max=100)
        grady = self.abs_sobel_thresh(
            orient='y', thresh_min=20, thresh_max=100)
        mag_binary = self.mag_thresh(sobel_kernel=3, mag_thresh=(20, 100))
        dir_binary = self.dir_threshold(
            sobel_kernel=3, thresh=(0.7, 1.3))

        gradient_binary = np.zeros_like(dir_binary)
        gradient_binary[((gradx == 1) & (grady == 1)) | (
            (mag_binary == 1) & (dir_binary == 1))] = 1

        color_binary = self.color()

        color_gradient_combined = np.zeros_like(dir_binary)
        color_gradient_combined[(gradient_binary == 1)
                                | (color_binary == 1)] = 1

        return color_gradient_combined
