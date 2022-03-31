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
    num_of_max_iter = 200
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
    colors = [2, 4, 8, 16, 23]
    fig, axs = plt.subplots(2, 3, figsize=(7, 4), constrained_layout=True, sharex='all', sharey='all')
    plt.gray()
    fontdict = {'fontsize': 12,
                'fontweight': 4}
    for ax in axs.flat:
        if i == len(colors):
            ax.imshow(img)
            ax.set_title("Original image", fontdict=fontdict)
        else:
            q, err = quantizeImage(img, colors[i], 20)
            ax.imshow(q[i])
            ax.set_title("Quantization of %d" % colors[i], fontdict=fontdict)
            i += 1
    plt.show()



def main():
    print("ID:", myID())
    img_path = 'beach.jpg'

    # Basic read and display
    imDisplay(img_path, LOAD_GRAY_SCALE)
    imDisplay(img_path, LOAD_RGB)
    #
    # # Convert Color spaces
    img = imReadAndConvert(img_path, LOAD_RGB)
    yiq_img = transformRGB2YIQ(img)
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(yiq_img)
    plt.show()

    # # Image histEq
    histEqDemo(img_path, LOAD_GRAY_SCALE)
    histEqDemo(img_path, LOAD_RGB)

    # Image Quantization
    quantDemo(img_path, LOAD_GRAY_SCALE)
    quantDemo(img_path, LOAD_RGB)

    # Gamma
    # gammaDisplay(img_path, LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
