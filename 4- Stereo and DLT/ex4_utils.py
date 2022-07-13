import math

import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from PIL import Image
from scipy.ndimage import filters


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    max_r = disp_range[1]  # max disparity range
    x, y = img_l.shape[0], img_l.shape[1]
    D_map = np.zeros((x, y, max_r))  # initialize disparity map

    # normalize the images by subtracting the mean.
    left_img = img_l - filters.uniform_filter(img_l, k_size)
    right_img = img_r - filters.uniform_filter(img_r, k_size)

    for curr_dist in range(disp_range[0], disp_range[1]):  # loop over disparities
        # move left img towards right img by dist
        steps = curr_dist + disp_range[0]
        L_img_to_R_img = np.roll(left_img, -steps)
        # compute the norm correlation
        sigma_l = L_img_to_R_img / filters.uniform_filter(np.square(L_img_to_R_img), k_size)
        sigma_r = right_img / filters.uniform_filter(np.square(right_img), k_size)
        SSD_sigma = filters.uniform_filter(np.square(sigma_l - sigma_r), k_size)

        # update the disparity map with the norm correlation of the current disparity.
        D_map[:, :, curr_dist] = np.square(SSD_sigma)
    return D_map.argmin(axis=2)


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorrelation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    x, y = img_l.shape[0], img_l.shape[1]
    D_map = np.zeros((x, y, disp_range[1]))
    # normalize the images by subtracting the mean.
    left_img = img_l - filters.uniform_filter(img_l, k_size)
    right_img = img_r - filters.uniform_filter(img_r, k_size)

    for curr_dist in range(disp_range[0], disp_range[1]):  # loop over disparities
        # move left img towards right img by dist
        steps = curr_dist + disp_range[0]
        L_img_to_R_img = np.roll(left_img, -steps)
        # compute the norm correlation
        sigma_l = filters.uniform_filter(np.square(L_img_to_R_img), k_size)
        sigma_r = filters.uniform_filter(np.square(right_img), k_size)
        NC_sigma = filters.uniform_filter(L_img_to_R_img * right_img, k_size)

        # update the disparity map with the norm correlation
        D_map[:, :, curr_dist] = NC_sigma / np.sqrt(sigma_l * sigma_r)
    return D_map.argmax(axis=2)


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ key points locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ key points locations (x,y) on the destination image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    loc = 0
    MAT = np.zeros((8, 9))
    for i in range(src_pnt.shape[0]):
        x, y = src_pnt[i, 0], src_pnt[i, 1]  # get the x,y coordinates of the src_pnt
        dx, dy = dst_pnt[i, 0], dst_pnt[i, 1]  # get the x,y coordinates of the dst_pnt
        MAT[loc, :] = [x, y, 1, 0, 0, 0, -dx * x, -dx * y, -dx]
        MAT[loc + 1, :] = [0, 0, 0, x, y, 1, -dy * x, -dy * y, -dy]
        loc += 2
    # compute the homography matrix
    ansV = np.linalg.svd(MAT)[2]  # get the last column of the matrix
    Homography = ansV[-1, :] / ansV[-1, -1]  # normalize the last column
    Homography = Homography.reshape(3, 3)

    # compute the error between the transformed points to their destination (matched) points
    start = np.vstack((src_pnt.T, np.ones(src_pnt.shape[0])))
    end = np.vstack((dst_pnt.T, np.ones(dst_pnt.shape[0])))
    err = np.sqrt(np.sum(Homography.dot(start) / Homography.dot(start)[-1] - end) ** 2)
    return Homography, err


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """
    try:
        dst_p = []
        fig1 = plt.figure()

        def onclick_1(event):
            x = event.xdata
            y = event.ydata
            print("Loc: {:.0f},{:.0f}".format(x, y))

            plt.plot(x, y, '*r')
            dst_p.append([x, y])

            if len(dst_p) == 4:
                plt.close()
            plt.show()

        # display image 1
        cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
        plt.imshow(dst_img)
        plt.show()
        dst_p = np.array(dst_p)

        ##### Your Code Here ######

        src_p = []
        fig2 = plt.figure()

        # same as onclick_1, but with the source image instead of the dest
        def onclick_2(event):
            x = event.xdata
            y = event.ydata
            print("Loc: {:.0f},{:.0f}".format(x, y))

            plt.plot(x, y, '*r')
            src_p.append([x, y])

            if len(src_p) == 4:
                plt.close()
            plt.show()

        # display image 2, same operations as display image 1
        cid = fig2.canvas.mpl_connect('button_press_event', onclick_2)
        plt.imshow(src_img)
        plt.show()
        src_p = np.array(src_p)

        cv_Homography = cv2.findHomography(src_p, dst_p)[0]

        warped_img = np.zeros_like(dst_img)
        for i in range(src_img.shape[0]):
            for j in range(src_img.shape[1]):
                # current pixel location
                curr_area = np.array([j, i, 1])
                # dot product of the current pixel location and the homography matrix
                curr_dot = np.dot(cv_Homography, curr_area)
                # normalize the dot product
                normed_y = int(curr_dot[0] / curr_dot[curr_dot.shape[0] - 1])
                normed_x = int(curr_dot[1] / curr_dot[curr_dot.shape[0] - 1])
                # if the normalized dot product is within the bounds of the destination image:
                # then copy the source image pixel to the destination image pixel location
                # in the warped image matrix (warped_img) using the normalized dot product.
                warped_img[normed_x, normed_y] = src_img[i, j]

        # create a mask of the warped image.
        mask = warped_img == 0
        # create a new image with the same dimensions as the warped image.
        # This will be the final image that we will display.
        # This image will be the destination image with the source image pasted on top.
        im2display = dst_img * mask + (1 - mask) * warped_img  # masking the destination image.
        plt.imshow(im2display)
        plt.show()

    except Exception as e:
        print(e)



