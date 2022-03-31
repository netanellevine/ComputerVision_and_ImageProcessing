from ex1_utils import *
from gamma import gammaDisplay
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2 as cv

import time


def histEqDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    imgeq, histOrg, histEq = hsitogramEqualize(img)

    # Display cumsum
    cumsum = np.cumsum(histOrg)
    cumsumEq = np.cumsum(histEq)
    plt.gray()
    plt.plot(range(256), cumsum, 'r')
    plt.plot(range(256), cumsumEq, 'g')

    # Display the images
    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(imgeq)
    plt.show()


def quantDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    st = time.time()
    num2quant = 3
    num_of_max_iter = 20
    img_lst, err_lst = quantizeImage(img, num2quant, num_of_max_iter)

    print("Time:%.2f" % (time.time() - st))
    print("Error 0:\t %.6f" % err_lst[0])
    print("Error last:\t %.6f" % err_lst[-1])
    i = err_lst.index(min(err_lst))
    print(i, len(err_lst))
    print(err_lst)
    plt.gray()

    ax1 = plt.subplot(2, 2, 1, frameon=False)
    plt.tick_params('both', labelbottom=False)
    plt.imshow(img)
    plt.title("Original image")
    # plt.figure()

    ax2 = plt.subplot(2, 2, 2, frameon=False)
    plt.tick_params('both', labelbottom=False)
    plt.imshow(img_lst[0])
    plt.title("First iteration")
    # plt.figure()

    ax3 = plt.subplot(2, 2, 3, frameon=False)
    plt.tick_params('both', labelbottom=False)
    plt.imshow(img_lst[-1])
    plt.title("Last iteration")
    # plt.figure()

    plt.subplot(2, 2, 4)
    plt.plot(err_lst, 'r')
    plt.title("MSE - Graph")

    title = "Quantize image from 256 to: {}, Max iterations: {}, actual: {}\n".format(num2quant, num_of_max_iter, len(err_lst))
    plt.suptitle(title)
    plt.show()

    # fig, (q2, q4, q8, q32, q64, qOrig) = plt.subplots(6, 1)
    # fig.suptitle("Quantize")
    # quantized = (q2, q4, q8, q32, q64, qOrig)
    # qi = [2, 4, 8, 16, 32, 64, 256]
    # i = 0
    # for qq in quantized:
    #     q, err = quantizeImage(img, qi[i], 50)
    #     qq.imshow(q[-1])
    #     plt.title("Quantization of %d" % qi[i])
    #     i += 1
    # q, err = quantizeImage(img, qi[i], 50)
    # q2.imshow(q[-1])

    # plt.show()


def main():
    print("ID:", myID())
    img_path = 'beach.jpg'

    # Basic read and display
    # imDisplay(img_path, LOAD_GRAY_SCALE)
    # imDisplay(img_path, LOAD_RGB)
    #
    # # Convert Color spaces
    # img = imReadAndConvert(img_path, LOAD_RGB)
    # yiq_img = transformRGB2YIQ(img)
    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(img)
    # ax[1].imshow(yiq_img)
    # plt.show()
    #
    # # Image histEq
    # histEqDemo(img_path, LOAD_GRAY_SCALE)
    # histEqDemo(img_path, LOAD_RGB)

    # Image Quantization
    quantDemo(img_path, LOAD_GRAY_SCALE)
    quantDemo(img_path, LOAD_RGB)

    # Gamma
    # gammaDisplay(img_path, LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
