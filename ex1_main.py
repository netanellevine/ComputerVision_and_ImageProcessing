from matplotlib import rcParams

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

    plt.subplot(2, 2, 1, frameon=False)
    plt.tick_params('x', labelbottom=False)
    plt.tick_params('y', labelbottom=False)
    plt.imshow(img)
    plt.title("Before")

    plt.subplot(2, 2, 2, frameon=False)
    plt.tick_params('x', labelbottom=False)
    plt.tick_params('y', labelbottom=False)
    plt.imshow(imgeq)
    plt.title("After")

    plt.subplot(2, 2, (3, 4))
    plt.plot(range(256), cumsum, 'r', label='Original')
    plt.plot(range(256), cumsumEq, 'g', label='Equalized')
    plt.xlabel("Intensity level", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.legend(loc='lower right', fontsize='11')

    title = "Histogram Equalization"
    plt.suptitle(title, fontsize=19)
    plt.show()


def quantDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    st = time.time()
    num2quant = 4
    num_of_max_iter = 100
    img_lst, err_lst = quantizeImage(img, num2quant, num_of_max_iter)

    print("Time:%.2f" % (time.time() - st))
    print("Error 0:\t %.6f" % err_lst[0])
    print("Error last:\t %.6f" % err_lst[-1])
    i = err_lst.index(min(err_lst))
    print(i, len(err_lst))
    # print(err_lst)
    plt.gray()

    ax1 = plt.subplot(2, 2, 1, frameon=False)
    plt.tick_params('x', labelbottom=False)
    plt.tick_params('y', labelbottom=False)
    plt.imshow(img)
    plt.title("Original image")

    ax2 = plt.subplot(2, 2, 2, frameon=False)
    plt.tick_params('x', labelbottom=False)
    plt.tick_params('y', labelbottom=False)
    plt.imshow(img_lst[0])
    plt.title("First iteration")

    ax3 = plt.subplot(2, 2, 3, frameon=False)
    plt.tick_params('x', labelbottom=False)
    plt.tick_params('y', labelbottom=False)
    plt.imshow(img_lst[-1])
    plt.title("Last iteration")

    plt.subplot(2, 2, 4)
    plt.plot(err_lst, 'r')
    plt.title("MSE - Graph")

    title = f'Quantize image from 256 to: {num2quant}, Max iterations: {num_of_max_iter}, actual: {len(err_lst)}\n'
    plt.suptitle(title, fontsize=14, color='b')
    plt.show()

    i = 0
    colors = [2, 4, 8, 16, 32]
    fig, axs = plt.subplots(2, 3, figsize=(7, 4), constrained_layout=True, sharex='all', sharey='all')
    plt.gray()
    fontdict = {'fontsize': 12,
                'fontweight': 4}
    for ax in axs.flat:
        if i == len(colors):
            ax.imshow(img)
            ax.set_title("Original image", fontdict=fontdict)
        else:
            q, err = quantizeImage(img, colors[i], 100)
            ax.imshow(q[i])
            ax.set_title("Quantization of %d" % colors[i], fontdict=fontdict)
            i += 1
    plt.show()


def main():
    print("ID:", myID())
    img_path = 'images/uluru.jpeg'

    # Basic read and display
    # imDisplay(img_path, LOAD_GRAY_SCALE)
    # imDisplay(img_path, LOAD_RGB)

    # Convert Color spaces
    # rgb_img = imReadAndConvert(img_path, LOAD_RGB)
    # gray_img = imReadAndConvert(img_path, LOAD_GRAY_SCALE)
    # yiq_img = transformRGB2YIQ(rgb_img)
    # i = 0
    # images = [gray_img, rgb_img, yiq_img]
    # spaces = ['Gray', 'RGB', 'YIQ']
    # fig, axs = plt.subplots(1, 3, figsize=(7, 3), constrained_layout=True, sharex='all', sharey='all')
    # plt.gray()
    # fontdict = {'fontsize': 12,
    #             'fontweight': 4}
    # for ax in axs.flat:
    #     ax.imshow(images[i])
    #     ax.set_title(f'{spaces[i]} Color Space', fontdict=fontdict)
    #     i += 1
    # title = f'Same image in several color spaces'
    # plt.suptitle(title, fontsize=18, fontweight=6)
    # plt.show()

    # # Image histEq
    # histEqDemo(img_path, LOAD_GRAY_SCALE)
    # histEqDemo(img_path, LOAD_RGB)

    # Image Quantization
    # quantDemo(img_path, LOAD_GRAY_SCALE)
    # quantDemo(img_path, LOAD_RGB)

    # Gamma
    # gammaDisplay(img_path, LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
