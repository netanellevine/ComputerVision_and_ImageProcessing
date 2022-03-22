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
import sys
from typing import List
from cv2 import cv2 as cv
import numpy as np
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
RGB2YIQ_mat = np.array([0.299, 0.587, 0.114, 0.596, -0.275, -0.321, 0.212, -0.523, 0.311]).reshape(3, 3)


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 312512619


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = None
    if representation == 1:
        img = cv.imread(filename, IMREAD_GRAYSCALE)
    elif representation == 2:
        img = cv.imread(filename, IMREAD_COLOR)
    if img is None:
        sys.exit("Could not read the image.")
    img = img.astype(np.float)
    norm_img = NormalizeData(img)
    return norm_img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    cv.imshow(filename, img)
    cv.waitKey(0)


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # RGB2YIQ_mat = np.array([0.299, 0.587, 0.114, 0.596, -0.275, -0.321, 0.212, -0.523, 0.311]).reshape(3, 3)
    orig_shape = imgRGB.shape
    imgRGB = imgRGB.reshape(-1, 3)
    YIQ_img = imgRGB.dot(RGB2YIQ_mat).reshape(orig_shape)
    return YIQ_img


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    orig_shape = imgYIQ.shape
    imgYIQ = imgYIQ.reshape(-1, 3)
    YIQ2RGB_mat = np.linalg.inv(RGB2YIQ_mat)
    RGB_img = imgYIQ.dot(YIQ2RGB_mat).reshape(orig_shape)
    return RGB_img


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))