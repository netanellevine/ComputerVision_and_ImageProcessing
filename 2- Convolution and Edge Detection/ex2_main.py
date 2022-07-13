from ex2_utils import *
import matplotlib.pyplot as plt
from cv2 import cv2 as cv2
import time


def MSE(a: np.ndarray, b: np.ndarray) -> float:
    return np.square(a - b).mean()


def MAE(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(MSE(a, b)).mean()


def conv1Demo():
    signal = np.array([1.1, 1, 3, 4, 5, 6, 2, 1])
    kernel = np.array([1, 2, 2, 1])

    signal = np.array([1, 2, 2, 1])
    kernel = np.array([1, 1])

    sig_conv = conv1D(signal, kernel).astype(int)

    print("Signal:\t{}".format(signal))
    print("Numpy:\t{}".format(np.convolve(signal, kernel, 'full')))
    print("Mine:\t{}".format(sig_conv))


def conv2Demo():
    img = cv2.imread('input/beach.jpeg', cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5, 5))
    kernel = kernel / kernel.sum()
    c_img = conv2D(img, kernel) / 255
    cv_img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE) / 255

    print("MSE: {}".format(255 * MSE(c_img, cv_img)))
    print("Max Error: {}".format(np.abs(c_img - cv_img).max() * 255))

    f, ax = plt.subplots(1, 3)
    ax[0].imshow(c_img)
    ax[1].imshow(cv_img - c_img)
    ax[2].imshow(cv_img)
    plt.show()


def derivDemo():
    img = cv2.imread('input/beach.jpeg', cv2.IMREAD_GRAYSCALE) / 255
    ori, mag = convDerivative(img)

    f, ax = plt.subplots(1, 2)
    ax[0].set_title('Ori')
    ax[1].set_title('Mag')
    ax[0].imshow(ori)
    ax[1].imshow(mag)
    plt.show()

    v = np.array([[1, 0, -1]])
    X = cv2.filter2D(img, -1, v)
    Y = cv2.filter2D(img, -1, v.T)

    ori = np.arctan2(Y, X).astype(np.float64)
    mag = np.sqrt(X ** 2 + Y ** 2).astype(np.float64)

    f, ax = plt.subplots(1, 2)
    ax[0].set_title('Ori')
    ax[1].set_title('Mag')
    ax[0].imshow(ori)
    ax[1].imshow(mag)
    plt.show()


def blurDemo():
    img = cv2.imread('input/beach.jpeg', cv2.IMREAD_GRAYSCALE) / 255
    k_size = 5
    b1 = blurImage1(img, k_size)
    b2 = blurImage2(img, k_size)

    print("Blurring MSE:{:.6f}".format(np.sqrt(np.power(b1 - b2, 2).mean())))

    f, ax = plt.subplots(1, 3)
    ax[0].imshow(b1)
    ax[1].imshow(b1 - b2)
    ax[2].imshow(b2)
    plt.show()


def edgeDemoSimple():
    img = cv2.imread('input/codeMonkey.jpeg', cv2.IMREAD_GRAYSCALE) / 255
    img = cv2.resize(img, (0, 0), fx=.25, fy=.25)
    edge_matrix = edgeDetectionZeroCrossingSimple(img)

    f, ax = plt.subplots(1, 2)
    ax[0].set_title("Ori")
    ax[1].set_title("Edge")
    ax[0].imshow(img)
    ax[1].imshow(edge_matrix)
    plt.show()


def edgeDemoLOG():
    img = cv2.imread('input/boxMan.jpeg', cv2.IMREAD_GRAYSCALE) / 255
    img = cv2.resize(img, (0, 0), fx=.25, fy=.25)
    edge_matrix = edgeDetectionZeroCrossingLOG(img)

    f, ax = plt.subplots(1, 2)
    ax[0].set_title("Ori")
    ax[1].set_title("Edge")
    ax[0].imshow(img)
    ax[1].imshow(edge_matrix)
    plt.show()


def edgeDemo():
    edgeDemoSimple()
    edgeDemoLOG()


def houghDemo():
    img = cv2.imread('input/coins.jpeg', cv2.IMREAD_GRAYSCALE) / 255
    min_r, max_r = 50, 100

    # # TEST WITH YOUR IMPLEMENT ONLY
    # img = cv2.imread('input/pool_balls.jpg', cv2.IMREAD_GRAYSCALE) / 255
    # min_r, max_r = 10, 20

    # Mine
    st = time.time()
    hough_rings = houghCircle(img, min_r, max_r)
    print("Hough Time[Mine]: {:.3f} sec".format(time.time() - st))
    # OpenCV
    st = time.time()
    cv2_cir = cv2.HoughCircles((img * 255).astype(np.uint8), cv2.HOUGH_GRADIENT, 1, minDist=30, param1=500,
                               param2=80, minRadius=min_r, maxRadius=max_r)
    print("Hough Time[CV]: {:.3f} sec".format(time.time() - st))

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    # Mine
    for c in hough_rings:
        circle1 = plt.Circle((c[0], c[1]), c[2], color='r', fill=False, linewidth=3)
        ax.add_artist(circle1)
    plt.show()
    # OpenCV
    for c in cv2_cir[0]:
        circle1 = plt.Circle((c[0], c[1]), c[2], color='g', fill=False, linewidth=2)
        ax.add_artist(circle1)
    plt.show()


def biliteralFilterDemo():
    img = cv2.imread('input/boxMan.jpeg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("original_image_grayscale.jpg", img)

    filtered_image_CV, filtered_image_my = bilateral_filter_implement(img, 9, 8.0, 1.0)
    cv2.imwrite("filtered_image_OpenCV.jpg", filtered_image_CV)
    cv2.imwrite("filtered_image_my.jpg", filtered_image_my)

    print(f"MSE: {MSE(filtered_image_my, filtered_image_CV)}")
    print(f"Max Error: {np.abs(filtered_image_my - filtered_image_CV).max()}")


def main():
    print("ID:", myID())
    conv1Demo()
    conv2Demo()
    derivDemo()
    blurDemo()
    edgeDemo()
    houghDemo()
    biliteralFilterDemo()


if __name__ == '__main__':
    main()
