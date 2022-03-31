"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import ex1_utils
from ex1_utils import LOAD_GRAY_SCALE
import sys
from typing import List
from cv2 import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE

gamma_slider_max = 200
title_window = 'Gamma Correction'
img = 0
# curr_gamma = 0
# trackbar_name = 'Gamma: %d' % curr_gamma


def on_trackbar(val: int):
    # global curr_gamma, trackbar_name
    # trackbar_name = 'Gamma: %d' % val
    # curr_gamma = val
    gamma = float(val) / 100
    invGamma = 1000 if gamma == 0 else 1.0 / gamma
    max_ = 255
    gammaTable = np.array([((i / float(max_)) ** invGamma) * max_
                           for i in np.arange(0, max_ + 1)]).astype("uint8")
    img_ = cv.LUT(img, gammaTable)
    scaled_img = cv.resize(img_, 960, 540)
    cv.imshow(title_window, scaled_img)


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img
    if rep == 1:
        img = cv.imread(img_path, 2)
    else:  # rep = LOAD_RGB
        img = cv.imread(img_path, 1)

    cv.namedWindow(title_window)
    trackbar_name = 'Gamma:'
    cv.createTrackbar(trackbar_name, title_window, 0, gamma_slider_max, on_trackbar)
    # Show some stuff
    on_trackbar(100)
    # Wait until user press some key
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)


def main():
    gammaDisplay('bac_con.png', 1)


if __name__ == '__main__':
    main()
