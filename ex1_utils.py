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
import math
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
    norm_img = normalizeData(img)
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


# def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
#     """
#         Equalizes the histogram of an image
#         :param imgOrig: Original Histogram
#         :ret
#     """
#     original_shape = imgOrig.shape
#     color = False
#     if len(original_shape) == 3:  # it's RGB scale img
#         YIQimg = transformRGB2YIQ(imgOrig)
#         tmpMat = YIQimg[:, :, 0] * 255
#         color = True
#     else:  # it's Gray scale img
#         tmpMat = imgOrig * 255
#     histOrg = np.histogram(tmpMat, bins=np.arange(0, 256))  # image histogram
#     cumSum = np.cumsum(histOrg[0])  # imag cumSum
#     LUT = np.zeros(256)
#     all_pixels = original_shape[0] * original_shape[1]  # amount of all the pixels of the image
#     for i in range(255):
#         LUT[i] = math.ceil((cumSum[i] / all_pixels) * 255)
#     for i in range(255):
#         tmpMat = replaceIntensity(tmpMat, LUT[i], i)
#     # tmpMat2[]
#     histEq = np.histogram(tmpMat, bins=np.arange(0, 256))
#     if color:  # RGB img
#         YIQimg[:, :, 0] = normalizeData(tmpMat)
#         new_img = transformYIQ2RGB(YIQimg)
#         # new_img = normalizeData(new_img)
#         # YIQimg[:, :, 0] = tmpMat
#         # RGBimg = transformYIQ2RGB(YIQimg)
#         # new_img = normalizeData(RGBimg)
#     else:  # Gray img
#         new_img = normalizeData(tmpMat)
#     return new_img, histOrg[0], histEq[0]
def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    original_shape = imgOrig.shape
    color = False
    if len(original_shape) == 3:  # it's RGB scale img
        YIQimg = transformRGB2YIQ(imgOrig)
        tmpMat = np.round(YIQimg[:, :, 0] * 255)
        color = True
    else:  # it's Gray scale img
        tmpMat = np.round(imgOrig * 255)
    histOrg = calHist(tmpMat)  # image histogram
    cumSum = calCumSum(histOrg)  # image cumSum
    LUT = np.zeros(256)
    all_pixels = original_shape[0] * original_shape[1]  # amount of all the pixels of the image
    for i in range(255):
        LUT[i] = math.ceil((cumSum[i] / all_pixels) * 255)
    imEqualized = np.zeros(imgOrig.shape)
    if color:  # RGB img
        imEqualized = transformRGB2YIQ(imgOrig)
        imEqualized[:, :, 0] = normalizeData(imEqualized[:, :, 0])
        for row in range(len(imEqualized)):
            for col in range(len(imEqualized[0])):
                imEqualized[row][col][0] = LUT[int(np.round(imEqualized[row][col][0]))]
        imEqualized[:, :, 0] = imEqualized[:, :, 0] / 255
        imEqualized = transformYIQ2RGB(imEqualized)
    else:  # Gray img
        for row in range(len(imEqualized)):
            for col in range(len(imEqualized[0])):
                imEqualized[row][col] = LUT[int(np.round(imgOrig[row][col] * 255))]

    histEq = calHist(tmpMat)
    # if color:  # RGB img
    #     YIQimg[:, :, 0] = normalizeData(tmpMat)
    #
    #     new_img = transformYIQ2RGB(YIQimg)
        # new_img = normalizeData(new_img)
        # YIQimg[:, :, 0] = tmpMat
        # RGBimg = transformYIQ2RGB(YIQimg)
        # new_img = normalizeData(RGBimg)
    # else:  # Gray img
    #     new_img = normalizeData(tmpMat)
    return imEqualized, histOrg, histEq

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass


def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def replaceIntensity(orig: np.ndarray, new_intensity: int, old_intensity: int):
    original_shape = orig.shape
    orig = orig.flatten()
    for i in range(len(orig)):
        if orig[i] == old_intensity:
            orig[i] = new_intensity
    orig = orig.reshape(original_shape)
    return orig


def calHist(img: np.ndarray) -> np.ndarray:
    img_flat = img.flatten()
    hist = np.zeros(256)
    for pix in img_flat:
        if round(pix) < 256:
            pix = round(pix)
        else:
            pix = 255
        hist[pix] += 1
    return hist


def calCumSum(arr: np.array) -> np.ndarray:
    cum_sum = np.zeros_like(arr)
    cum_sum[0] = arr[0]
    arr_len = len(arr)
    for idx in range(1, arr_len):
        cum_sum[idx] = arr[idx] + cum_sum[idx - 1]
    return cum_sum

