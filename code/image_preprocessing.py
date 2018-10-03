import cv2
import numpy as np


def imagePreprocessing(img):
    # convert to grayscale image
    if (len(img.shape) == 3):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # normalize values between [0, 1]
    img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # scale input image
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    return img
