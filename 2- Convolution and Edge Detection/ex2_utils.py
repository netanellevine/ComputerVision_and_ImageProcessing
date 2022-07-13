import math
import numpy as np
from cv2 import cv2 as cv


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 312512619


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    """
    The formula for calculating 1D convolution.
    ______________________________________________
    ||##########################################||
    ||#                    _____               #||
    ||# h[j] = (f*g)[j] =  \\ f[j-i]*g[i]      #||
    ||#                    //___               #||
    ||#             -inf <= i <= inf           #||
    ||##########################################||    
    ||__________________________________________||          
    """

    g = np.flip(k_size)
    f = np.concatenate([np.zeros(g.size - 1), in_signal, np.zeros(g.size - 1)])
    h = np.zeros(in_signal.size + g.size - 1)
    for j, i in enumerate(range(g.size, f.size + 1)):
        curr_h = (f[j:i] * g)
        # print(f[j:i], " * ",  * g, " = ", (f[j:i] * g))
        h[j] = curr_h.sum()
    return h


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    mat = in_image
    kernel = np.flip(kernel)  # first step flip the kernel horizontally & vertically.
    ker_shape = kernel.shape[0]
    img_height, img_weight = in_image.shape
    ans_mat = np.zeros(in_image.shape)
    num2pad = math.floor(ker_shape / 2)  # choose how much to pad according to the kernel shape.
    mat = np.pad(mat, pad_width=num2pad, mode='edge')  # choose which padding to use -> here I chose by edge.
    for row in range(img_height):
        for column in range(img_weight):
            # at each iteration create a temp matrix with the same shape as the kernel.
            curr_mat = mat[row:ker_shape + row, column:ker_shape + column]
            # multiply the two matrices sum them up and round.
            new_val = ((curr_mat * kernel).sum()).round()
            # the new value is the intensity level after convulsed.
            ans_mat[row][column] = new_val
    return ans_mat


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale image
    :return: (directions, magnitude)
    """
    # first we'll create a kernel that suites for computing the gradient
    kernel = np.array([[1, 0, -1]])
    # calculate the X derivative with conv2D
    x_derivative = cv.filter2D(in_image, -1, kernel)
    kernel = kernel.T  # in order to compute the y derivative we need to transpose the kernel
    # calculate the Y derivative with conv2D
    y_derivative = cv.filter2D(in_image, -1, kernel)
    """
    Magnitude = |∇f| =  √((∂f/∂x)² + (∂f/∂y)²)
    Direction = α = arc-tan((∂f/∂y) / (∂f/∂x))
    """
    magnitude = np.sqrt((x_derivative ** 2 + y_derivative ** 2)).astype(np.float64)
    directions = np.arctan2(y_derivative, x_derivative).astype(np.float64)
    return directions, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    gaussian = np.exp(-0.5 * (np.square(np.linspace(-(k_size - 1) / 2, (k_size - 1) / 2, k_size))))
    # create a gaussian kernel
    gaussian_kernel = np.outer(gaussian, gaussian)
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    return cv.filter2D(in_image, -1, gaussian_kernel, borderType=cv.BORDER_REPLICATE)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    if k_size % 2 == 0:
        return "The kernel size must be odd"
    sigma = (k_size - 1) / 6
    gaussian_kernel = cv.getGaussianKernel(k_size, sigma)  # create a gaussian kernel
    return cv.filter2D(in_image, -1, gaussian_kernel, borderType=cv.BORDER_REPLICATE)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    # I did the edge detection LOG
    return img


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    after_smooth = cv.GaussianBlur(img, (5, 5), 0)
    # better kernel then the one with 4 in the middle.
    laplacian_kernel = np.array([[-1, -1, -1],
                                 [-1, 8, -1],
                                 [-1, -1, -1]])
    laplacian = cv.filter2D(after_smooth, -1, laplacian_kernel, borderType=cv.BORDER_REPLICATE)
    zero_crossing = np.zeros(img.shape)
    # Counting the positive and negative pixels in the neighborhood
    for i in range(1, laplacian.shape[0] - 1):
        for j in range(1, laplacian.shape[1] - 1):
            neg_count = pos_count = 0
            # neighborhoods
            neighbours = [laplacian[i + 1, j - 1], laplacian[i + 1, j], laplacian[i + 1, j + 1], laplacian[i, j - 1], laplacian[i, j + 1],
                          laplacian[i - 1, j - 1], laplacian[i - 1, j], laplacian[i - 1, j + 1]]
            max_neighborhood = max(neighbours)
            min_neighborhood = min(neighbours)
            # checking for positive and negative
            for g in neighbours:
                if g > 0:
                    pos_count += 1
                if g < 0:
                    neg_count += 1
            z_c = ((neg_count > 0) and (pos_count > 0))
            # changing the pixel value with the maximum difference in the neighborhood
            if z_c:
                if img[i, j] > 0:
                    zero_crossing[i, j] = laplacian[i, j] + np.abs(min_neighborhood)
                if img[i, j] < 0:
                    zero_crossing[i, j] = np.abs(laplacian[i, j]) + max_neighborhood
    return zero_crossing


def create_radius_step(diff: int) -> int:
    if diff < 60:
        return 1
    elif diff < 120:
        return 2
    elif diff < 180:
        return 3
    elif diff < 240:
        return 4
    elif diff < 300:
        return 5
    elif diff < 360:
        return 6
    else:
        return 8


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """
    circles_list = []
    threshold = 2  # after some tests and online search this is the threshold that suited to most of the inputs.
    # In case the img is normalized between zero and one -> normalize between 0 and 255.
    if img.max() <= 1:
        img = (cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)).astype('uint8')

    radius_by_shape = min(img.shape[0], img.shape[1]) // 2
    max_radius = min(radius_by_shape, max_radius)
    accumulator = np.zeros((len(img), len(img[0]), max_radius + 1))
    # calculating the sobel
    x_derivative = cv.Sobel(img, cv.CV_64F, 1, 0, threshold)
    y_derivative = cv.Sobel(img, cv.CV_64F, 0, 1, threshold)
    # calculating the direction(angle).
    direction = np.arctan2(y_derivative, x_derivative)
    direction = np.radians(direction * 180 / np.pi)

    # if the difference between the min radius to the max radius is too big to save some time we will increase the step.
    diff = max_radius - min_radius
    rad_step = create_radius_step(diff)

    # instead of looking at all the image will look only at the image edges.
    canny_edges = cv.Canny(img, 75, 150)
    for x in range(len(canny_edges)):
        for y in range(len(canny_edges[0])):
            # if this pixel is an edge
            if canny_edges[x][y] == 255:
                for rad in range(min_radius, max_radius + 1, rad_step):
                    angle = direction[x, y] - np.pi / 2
                    x1, x2 = (x - rad * np.cos(angle)).astype(np.int32), (x + rad * np.cos(angle)).astype(np.int32)
                    y1, y2 = (y + rad * np.sin(angle)).astype(np.int32), (y - rad * np.sin(angle)).astype(np.int32)
                    if 0 < x1 < len(accumulator) and 0 < y1 < len(accumulator[0]):
                        accumulator[x1, y1, rad] += 1
                    if 0 < x2 < len(accumulator) and 0 < y2 < len(accumulator[0]):
                        accumulator[x2, y2, rad] += 1

    threshold = np.multiply(np.max(accumulator), 1 / 2) + 1  # updating the threshold
    x, y, rad = np.where(accumulator >= threshold)  # getting the circles' that are after the threshold
    circles_list.extend((y[i], x[i], rad[i]) for i in range(len(x)) if x[i] != 0 or y[i] != 0 or rad[i] != 0)
    return circles_list


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    # opencv implementation
    opencv = cv.bilateralFilter(src=in_image, d=k_size, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    k_size //= 2
    # padding the image by half of the kernel size from each side
    padded_img = cv.copyMakeBorder(in_image, top=k_size, bottom=k_size, left=k_size, right=k_size, borderType=cv.BORDER_REFLECT_101).astype(int)
    outcome = in_image.copy()
    for y in range(k_size, padded_img.shape[0] - k_size):
        for x in range(k_size, padded_img.shape[1] - k_size):
            pivot = padded_img[y, x]
            neighbor_hood = padded_img[y - k_size:y + k_size + 1, x - k_size:x + k_size + 1]
            diff = pivot - neighbor_hood
            diff_gauss = np.exp(-np.power(diff, 2) / (2 * sigma_color))
            gaussian_kernel = cv.getGaussianKernel(2 * k_size + 1, sigma=sigma_space)
            gaussian = gaussian_kernel.dot(gaussian_kernel.T)
            # the formula is multiplying the distance and intensity difference
            formula = gaussian * diff_gauss
            answer = (formula * neighbor_hood) / formula.sum()
            outcome[y - k_size, x - k_size] = round(answer.sum())
    return opencv, outcome

