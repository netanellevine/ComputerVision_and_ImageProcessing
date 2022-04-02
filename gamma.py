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
from cv2 import cv2 as cv
import numpy as np
title_window = 'Gamma Correction'
trackbar_name = 'Gamma:'
gamma_slider_max_val = 200
max_pix = 255
isColor = False
img = 0


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img
    if rep == 1:  # Gray img
        img = cv.imread(img_path, 2)
    else:  # RGB img
        img = cv.imread(img_path, 1)

    cv.namedWindow(title_window)
    cv.createTrackbar(trackbar_name, title_window, 100, gamma_slider_max_val, on_trackbar)
    # Show the trackbar
    on_trackbar(100)  # Start when the img is multiply by gamma = 1 == 100 a.k.a. the original img.
    # Wait until user press some key
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)


def on_trackbar(val: int):
    gamma = float(val) / 100
    inv_gamma = 1000 if gamma == 0 else 1.0 / gamma
    gammaMat = np.array([((i / float(max_pix)) ** inv_gamma) * max_pix
                         for i in np.arange(0, max_pix + 1)]).astype("uint8")
    img_after_gamma = cv.LUT(img, gammaMat)

    cv.imshow(title_window, img_after_gamma)


def main():
    gammaDisplay('images/sunrise.jpeg', 2)


if __name__ == '__main__':
    main()
