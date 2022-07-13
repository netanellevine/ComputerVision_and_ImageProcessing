import itertools
import math
import sys
from typing import List

import numpy as np
import cv2
import pygame
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    """
    return 312512619


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

def myOpticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                  win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    image_shape = im1.shape
    if len(image_shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        plt.gray()
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        plt.gray()
    if win_size % 2 == 0:
        return "win_size must be an odd number"
    half_win_size = win_size // 2

    kernel_x = np.array([[-1, 0, 1]])
    kernel_y = kernel_x.T

    Ix = cv2.filter2D(im2, -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im2, -1, kernel_y, borderType=cv2.BORDER_REPLICATE)
    It = im2 - im1

    original_points = []  # list for the original points.
    vec_per_point = []  # list for the OP vectors for each point in @original_points.

    # Here I will go over a blocks in the shape of (step_size x step_size)
    # and find the optical flow vector for each block.
    # optional start from @half_win_size + 1
    for row in range(step_size, image_shape[0] - half_win_size + 1, step_size):
        for col in range(step_size, image_shape[1] - half_win_size + 1, step_size):
            Ix_windowed = Ix[
                          row - half_win_size: row + half_win_size + 1,
                          col - half_win_size: col + half_win_size + 1,
                          ].flatten()
            Iy_windowed = Iy[
                          row - half_win_size: row + half_win_size + 1,
                          col - half_win_size: col + half_win_size + 1,
                          ].flatten()
            It_windowed = It[
                          row - half_win_size: row + half_win_size + 1,
                          col - half_win_size: col + half_win_size + 1,
                          ].flatten()
            A = np.vstack((Ix_windowed, Iy_windowed)).T  # A = [Ix, Iy]
            b = (A.T @ (-1 * It_windowed).T).reshape(2, 1)
            ATA = A.T @ A

            ATA_eig_vals = np.sort(np.linalg.eigvals(ATA))
            if ATA_eig_vals[0] <= 1 or ATA_eig_vals[1] / ATA_eig_vals[0] >= 100:
                # vec_per_point.append(np.array([0, 0]))
                # original_points.append([col, row])
                continue

            ATA_INV = np.linalg.inv(ATA)
            curr_vec = ATA_INV @ b
            original_points.append([col, row])
            vec_per_point.append([curr_vec[0, 0], curr_vec[1, 0]])

    return original_points, vec_per_point


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    original_points, vec_per_point = myOpticalFlow(im1, im2, step_size, win_size)
    return np.array(original_points), np.array(vec_per_point)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """

    # if the image is RGB, convert to gray
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # if the images don't have the same shape we will throw an error
    if img1.shape != img2.shape:
        raise Exception ("The images must be in the same size")

    # First, find the pyramids for @img1 and @img2.
    firstImgPyramid = gaussianPyr(img1, k)
    firstImgPyramid.reverse()
    secondImgPyramid = gaussianPyr(img2, k)
    secondImgPyramid.reverse()

    # find the OP for the first img
    original_points, vec_per_point = myOpticalFlow(firstImgPyramid[0], secondImgPyramid[0], stepSize, winSize)

    for i in range(1, k):
        orig_pyr_ind, vec_pyr_ind = myOpticalFlow(firstImgPyramid[i], secondImgPyramid[i], stepSize, winSize)
        for j in range(len(original_points)):
            original_points[j] = [element * 2 for element in original_points[j]]
            vec_per_point[j] = [element * 2 for element in vec_per_point[j]]

        # add the OP vectors for each of the pyramids.
        for pixel, uv_current in zip(orig_pyr_ind, vec_pyr_ind):
            if pixel not in original_points:
                original_points.append(pixel)
                vec_per_point.append(uv_current)
            else:
                vec_per_point[original_points.index(pixel)][0] += uv_current[0]
                vec_per_point[original_points.index(pixel)][1] += uv_current[1]

    ans = np.zeros(shape=(img1.shape[0], img1.shape[1], 2))
    # reshape to 3D array (X, Y, 2)
    for ind in range(len(original_points)):
        px = original_points[ind][1]
        py = original_points[ind][0]
        ans[px][py][0] = vec_per_point[ind][0]
        ans[px][py][1] = vec_per_point[ind][1]
    return ans


def OF(im1: np.ndarray, im2: np.ndarray, blockSize=11,
       maxCorners=5000, qualityLevel=0.00001, minDistance=1) -> (np.ndarray, np.ndarray):
    # this function find the OF by using openCV LK
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')
    features = dict(maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance, blockSize=blockSize)
    best_points = cv2.goodFeaturesToTrack(im1, mask=None, **features)
    movements = cv2.calcOpticalFlowPyrLK(im1, im2, best_points, None)[0]
    return movements, best_points


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def find_vec_of_transformMed(mat: np.ndarray):
    """
    This method calculates the median of the (u,v) vector from @mat.
    :param mat: array of the vectors of the OP
    :return: the median vector - (u,v)
    """
    U = []
    V = []
    for ind in range(len(mat)):
        U.append(mat[ind, 0, 0])
        V.append(mat[ind, 0, 1])
    med1 = np.median(U)
    med2 = np.median(V)
    return med1, med2


def find_vec_of_transform(img1: np.ndarray, img2: np.ndarray, mat: np.ndarray):
    """
    This function returns best vector that suits the translation between @img1 to @img2.
    :param img1: first image
    :param img2: second image
    :param mat: array of the vectors of the OP
    :return: vector - [u,v]
    """
    best_fit = float("inf")
    best_vec = 0
    for ind in range(len(mat)):
        tx = mat[ind, 0, 0]
        ty = mat[ind, 0, 0]
        t = np.array([[1, 0, tx],
                      [0, 1, ty],
                      [0, 0, 1]], dtype=np.float)
        curr_im2 = cv2.warpPerspective(img1, t, img1.shape[::-1])
        curr_fit = ((img2 - curr_im2) ** 2).sum()
        if curr_fit < best_fit:
            best_fit = curr_fit
            best_vec = [tx, ty]
        if curr_fit == 0:
            break

    return best_vec


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """

    movements, best_points = OF(im1, im2)
    # find the vector by checking all the @vec vectors.
    tx, ty = find_vec_of_transform(im1, im2, movements - best_points)

    # find the vector by calculating the median vector from @vec (different approach)
    # tx, ty = find_vec_of_transformMed(movements - best_points)

    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])


def bestAngle(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    This function go over all the possibilities for an angle between two images (0-359).
    :param img1: first image
    :param img2: second image
    :return: the best angle
    """
    best_angle = 0
    best_fit = float("inf")
    for angle in range(360):  # for every possible angle
        cos_value = math.cos(math.radians(angle))
        sin_value = math.cos(math.radians(angle))

        # The rotation transformation matrix.
        rotation_mat = np.array([[cos_value, -sin_value, 0], [sin_value, cos_value, 0], [0, 0, 1]], dtype=np.float32)

        curr_img2 = cv2.warpPerspective(img1, rotation_mat, img1.shape[::-1])
        curr_fit = mean_squared_error(img2, curr_img2)

        if curr_fit < best_fit:  # goal min MSE -> == 0.
            best_fit = curr_fit
            best_angle = angle
        if best_fit == 0:
            break

    return best_angle


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    best_angle = bestAngle(im1, im2)
    # after discovered the @best_angle we can create the rotation matrix.
    cos_value = math.cos(math.radians(best_angle))
    sin_value = math.sin(math.radians(best_angle))
    rotation_mat = np.array([[cos_value, -sin_value, 0], [sin_value, cos_value, 0], [0, 0, 1]], dtype=np.float32)

    # now we need to find out the (u,v) that will complete the transformation from rotation to rigid.
    # rigid = rotation X translation
    after_rotate_im2 = cv2.warpPerspective(im1, rotation_mat, im1.shape[::-1])
    translation_mat = findTranslationLK(after_rotate_im2, im2)

    return translation_mat @ rotation_mat


def findCorrelation(img1: np.ndarray, img2: np.ndarray):
    """
    This function looks for two points, one from @img1 and second from @img2.
    The two points are the ones with the highest correlation.
    :param img1: first image
    :param img2: second image
    :return: 2 points - x1, y1, x2, y2
    """
    # img1[img1 == 0] = np.float("inf")
    # img2[img2 == 0] = np.float("inf")
    img_shape = np.max(img1.shape) // 2
    im1FFT = np.fft.fft2(np.pad(img1, img_shape))
    im2FFT = np.fft.fft2(np.pad(img2, img_shape))
    prod = im1FFT * im2FFT.conj()
    res = np.fft.fftshift(np.fft.ifft2(prod))
    correlation = res.real[1 + img_shape:-img_shape + 1, 1 + img_shape:-img_shape + 1]
    p1y, p1x = np.unravel_index(np.argmax(correlation), correlation.shape)
    p2y, p2x = np.array(img2.shape) // 2
    return p1x, p1y, p2x, p2y


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    X1, Y1, X2, Y2 = findCorrelation(im1, im2)
    return np.array([[1, 0, (X2 - X1 - 1)], [0, 1, (Y2 - Y1 - 1)], [0, 0, 1]], dtype=np.float)


def getAngle(point1, point2):
    """
    This function calculate the angle between @point1 to @point2 by checking the intersection
    point of these points by creating two lines to the mass of center (0,0).
    :param point1: [x,y]
    :param point2: [x,y]
    :return: float - angle
    """
    vec1 = pygame.math.Vector2(point1[0] - 0, point1[1] - 0)
    vec2 = pygame.math.Vector2(point2[0] - 0, point2[1] - 0)
    return vec1.angle_to(vec2)


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    min_error = np.float('inf')
    best_rotation_mat = best_rotated_img = 0
    # for every angle between 0 and 359 we calculate the correlation.
    # and we find the best angle by checking the correlation.
    # and then we find the best rotated image by checking the correlation.
    for angle in range(360):
        rotation_mat = np.array([[math.cos(angle), -math.sin(angle), 0],
                      [math.sin(angle), math.cos(angle), 0],
                      [0, 0, 1]], dtype=np.float32)
        after_rotation = cv2.warpPerspective(im1, rotation_mat, im1.shape[::-1])  # rotating the image
        curr_mse = mean_squared_error(im2, after_rotation)  # calculating the error
        if curr_mse < min_error:
            min_error = curr_mse
            best_rotation_mat = rotation_mat
            best_rotated_img = after_rotation.copy()
        if curr_mse == 0:
            break

    translation = findTranslationCorr(best_rotated_img, im2)  # finding the translation from the rotated
    return translation @ best_rotation_mat  # combining the translation and the rotation.


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    warped_img = np.zeros(im2.shape)
    for row in range(1, im2.shape[0]):
        for col in range(1, im2.shape[1]):
            # we need to find the coordinates of the point in image 1
            # by using the inverse transformation.
            current_vec = np.array([row, col, 1]) @ np.linalg.inv(T)  # calculating the vector
            dx, dy = int(round(current_vec[0])), int(round(current_vec[1]))
            # if the point has a valid coordinates in image 1
            # we put the pixel in the warped image.
            if 0 <= dx < im1.shape[0] and 0 <= dy < im1.shape[1]:
                warped_img[row, col] = im1[dx, dy]  # putting the pixel in the warped image.

    return warped_img


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def get_sigma(k_size: int):
    return 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    kernel = cv2.getGaussianKernel(5, sigma=get_sigma(5))
    kernel = np.dot(kernel, kernel.T)
    gauss_pyramid = gaussianPyr(img, levels)
    gauss_pyramid.reverse()

    lap_pyramid = [gauss_pyramid[0]]
    for i in range(len(gauss_pyramid) - 1):
        # expand to next level
        expandedImg = (gaussExpand(gauss_pyramid[i], kernel))

        try:  # if we can add the two images without changing their size
            curr_layer = (cv2.subtract(gauss_pyramid[i + 1], expandedImg))
        except Exception:
            if len(expandedImg) != len(gauss_pyramid[i + 1]):
                expandedImg = expandedImg[0:len(gauss_pyramid[i + 1]), :]
            if len(expandedImg[0]) != len(gauss_pyramid[i + 1][0]):
                expandedImg = expandedImg[:, 0:len(gauss_pyramid[i + 1][0])]
            curr_layer = (cv2.subtract(gauss_pyramid[i + 1], expandedImg))

        lap_pyramid.append(curr_layer)
    lap_pyramid.reverse()
    return lap_pyramid


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resorts the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    kernel = cv2.getGaussianKernel(5, sigma=get_sigma(5))
    kernel = np.dot(kernel, kernel.T)
    output_img = lap_pyr[-1]
    for i in range(len(lap_pyr) - 1, 0, -1):
        # expand to next level
        expandedImg = gaussExpand(output_img, kernel)

        try:  # if we can add the two images without changing their size
            output_img = cv2.add(expandedImg, lap_pyr[i - 1])
        except Exception:
            if len(expandedImg) != len(lap_pyr[i - 1]):
                expandedImg = expandedImg[0:len(lap_pyr[i - 1]), :]
            if len(expandedImg[0]) != len(lap_pyr[i - 1][0]):
                expandedImg = expandedImg[:, 0:len(lap_pyr[i - 1][0])]
            output_img = cv2.add(expandedImg, lap_pyr[i - 1])

    return output_img


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    gauss_pyramid = [img]

    for _ in range(1, levels):
        curr_img = cv2.GaussianBlur(gauss_pyramid[-1], (5, 5), 0)
        curr_img = curr_img[::2, ::2]
        gauss_pyramid.append(curr_img)

    return gauss_pyramid


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    # check the img type and build output accordingly
    if len(img.shape) > 2:
        expanded_img = np.zeros((2 * len(img), 2 * len(img[0]), 3))
    else:
        expanded_img = np.zeros((2 * len(img), 2 * len(img[0])))
    # duplicate pixels
    expanded_img[::2, ::2] = img
    expanded_img[expanded_img < 0] = 0
    expanded_img = cv2.filter2D(expanded_img, -1, gs_k * 4, borderType=cv2.BORDER_REPLICATE)
    return expanded_img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    firstLaplacianPyr = laplaceianReduce(img_1, levels=levels)
    secondLaplacianPyr = laplaceianReduce(img_2, levels=levels)
    gaussPyrMask = gaussianPyr(mask, levels=levels)
    # after we got the pyramids for @img_1 and for @img_2 and for the mask we can blend
    # each level of those pyramids and create a new blended pyramid.
    blended_pyr = [gaussPyrMask[i] * firstLaplacianPyr[i] + (1 - gaussPyrMask[i]) * secondLaplacianPyr[i] for i in
                   range(len(gaussPyrMask))]

    # now we can expand and build our desired image from the pyramid.
    blended_img = laplaceianExpand(blended_pyr)
    naive_img = img_1.copy()
    naive_img[mask == 0] = img_2[mask == 0]

    return naive_img, blended_img
