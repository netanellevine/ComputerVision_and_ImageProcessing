from matplotlib import pyplot as plt
from ex3_utils import *
# from ex3T import *
# from ex3 import *
import time


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("---------------------------------------------------------------------------")
    print("LK Demo")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(np.float), img_2.astype(np.float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.2f}".format(et - st))
    print(np.median(uv, 0))
    print(np.mean(uv, 0))

    displayOpticalFlow(img_2, pts, uv)


def hierarchicalkDemo(img_path):
    print("---------------------------------------------------------------------------")
    print("Hierarchical LK Demo")
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(im1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float)
    im2 = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))

    ans = opticalFlowPyrLK(im1.astype(np.float), im2.astype(np.float), 4, 20, 5)

    pts = np.array([])
    uv = np.array([])
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            if ans[i][j][1] != 0 and ans[i][j][0] != 0:
                uv = np.append(uv, ans[i][j][0])
                uv = np.append(uv, ans[i][j][1])
                pts = np.append(pts, j)
                pts = np.append(pts, i)
    pts = pts.reshape(int(pts.shape[0] / 2), 2)
    uv = uv.reshape(int(uv.shape[0] / 2), 2)
    print(np.median(uv, 0))
    print(np.mean(uv, 0))
    displayOpticalFlow(im2, pts, uv)


def compareLK(img_path):
    print("---------------------------------------------------------------------------")
    print("Compare LK & Hierarchical LK")

    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(im1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -0.2],
                  [0, 1, -0.1],
                  [0, 0, 1]], dtype=np.float)
    im2 = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))

    pts, uv = opticalFlow(im1.astype(np.float), im2.astype(np.float), step_size=20, win_size=5)

    ans = opticalFlowPyrLK(im1.astype(np.float), im2.astype(np.float), 4, 20, 5)
    ptspyr = np.array([])
    uvpyr = np.array([])
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            if ans[i][j][1] != 0 and ans[i][j][0] != 0:
                uvpyr = np.append(uvpyr, ans[i][j][0])
                uvpyr = np.append(uvpyr, ans[i][j][1])
                ptspyr = np.append(ptspyr, j)
                ptspyr = np.append(ptspyr, i)
    ptspyr = ptspyr.reshape(int(ptspyr.shape[0] / 2), 2)
    uvpyr = uvpyr.reshape(int(uvpyr.shape[0] / 2), 2)
    if len(im2.shape) == 2:
        f, ax = plt.subplots(3, 1, figsize=(25, 35))
        # ax[0].set_title('reg LK', fontsize=20)
        ax[0].imshow(im2, cmap="gray")
        ax[0].quiver(pts[:, 0], pts[:, 1], uv[:, 0], uv[:, 1], color='r')
        # ax[1].set_title('Pyr LK', fontsize=20)
        ax[1].imshow(im2, cmap="gray")
        ax[1].quiver(ptspyr[:, 0], ptspyr[:, 1], uvpyr[:, 0], uvpyr[:, 1], color='r')
        # ax[2].set_title('overlap', fontsize=20)
        ax[2].imshow(im2, cmap="gray")
    else:
        f, ax = plt.subplots(3, 1, figsize=(25, 35))
        # ax[0].set_title('reg LK', loc='left', fontsize=35)
        ax[0].imshow(im2)
        ax[0].quiver(pts[:, 0], pts[:, 1], uv[:, 0], uv[:, 1], color='r')
        # ax[1].set_title('Pyr LK', fontsize=35)
        ax[1].imshow(im2)
        ax[1].quiver(ptspyr[:, 0], ptspyr[:, 1], uvpyr[:, 0], uvpyr[:, 1], color='r')
        # ax[2].set_title('overlap', fontsize=35)
        ax[2].imshow(im2)

    ax[2].quiver(pts[:, 0], pts[:, 1], uv[:, 0], uv[:, 1], color='r')
    ax[2].quiver(ptspyr[:, 0], ptspyr[:, 1], uvpyr[:, 0], uvpyr[:, 1], color='y')
    f.tight_layout()
    plt.show()


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()
    print("---------------------------------------------------------------------------")
    print("Optical flow demo")
    # if len(img.shape)==2:
    #     plt.imshow(img, cmap='gray')
    #     plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')
    #     plt.show()
    # else:
    #     plt.imshow(img)
    #     plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')
    #     plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------

def translationlkdemo(img_path):
    print("---------------------------------------------------------------------------")
    print("Translation lk demo")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    orig_mat = np.array([[1, 0, -50],
                         [0, 1, -50],
                         [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, orig_mat, img_1.shape[::-1])
    cv2.imwrite('imTransA1.jpg', img_2)
    st = time.time()
    my_mat = findTranslationLK(img_1, img_2)
    et = time.time()
    print("Time: {:.2f}".format(et - st))
    print("my mat: \n", my_mat, "\n\n original mat: \n", orig_mat)
    my_warp = cv2.warpPerspective(img_1, my_mat, (img_1.shape[1], img_1.shape[0]))
    cv2.imwrite('imTransA2.jpg', my_warp)
    f, ax = plt.subplots(1, 3)
    ax[0].set_title('CV Translation')
    ax[0].imshow(img_2, cmap='gray')

    ax[1].set_title('My Translation')
    ax[1].imshow(my_warp, cmap='gray')

    ax[2].set_title('Difference')
    ax[2].imshow(img_2 - my_warp, cmap='gray')

    plt.show()
    print("MSE = ", MSE(my_warp, img_2))


def rigidlkdemo(img_path):
    print("---------------------------------------------------------------------------")
    print("Rigid lk demo")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    angle = 0.8

    rigid = np.array([[np.cos(angle), -np.sin(angle), -1],
                      [np.sin(angle), np.cos(angle), -1],
                      [0, 0, 1]], dtype=np.float32)

    img_2 = cv2.warpPerspective(img_1, rigid, img_1.shape[::-1])
    cv2.imwrite('imRigidA1.jpg', img_2)
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('CV Rigid')
    ax[0].imshow(img_2, cmap='gray')
    start = time.time()
    my_rigid = findRigidLK(img_1, img_2)
    end = time.time()
    print("Time: {:.2f}".format(end - start))
    print("my mat: \n", my_rigid, "\n\n original mat: \n", rigid)

    my_warp = cv2.warpPerspective(img_1, my_rigid, img_1.shape[::-1])
    cv2.imwrite('imRigidA2.jpg', my_warp)
    ax[1].set_title('My Rigid')
    ax[1].imshow(my_warp, cmap='gray')
    plt.show()


def translationcorrdemo(img_path):
    print("---------------------------------------------------------------------------")
    print("Translation corr demo")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    orig_mat = np.array([[1, 0, 35],
                         [0, 1, 80],
                         [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, orig_mat, img_1.shape[::-1])
    cv2.imwrite('imTransB1.jpg', img_2)
    st = time.time()
    my_mat = findTranslationCorr(img_1, img_2)
    et = time.time()
    print("Time: {:.2f}".format(et - st))
    print("my mat: \n", my_mat, "\n\n original mat: \n", orig_mat)
    my_warp = cv2.warpPerspective(img_1, my_mat, (img_1.shape[1], img_1.shape[0]))
    cv2.imwrite('imTransB2.jpg', my_warp)
    f, ax = plt.subplots(1, 3)
    ax[0].set_title('CV Translation')
    ax[0].imshow(img_2, cmap='gray')

    ax[1].set_title('My Translation')
    ax[1].imshow(my_warp, cmap='gray')

    ax[2].set_title('Difference')
    ax[2].imshow(img_2 - my_warp, cmap='gray')

    plt.show()
    print("MSE = ", MSE(my_warp, img_2))


def rigidcorrdemo(img_path):
    print("---------------------------------------------------------------------------")
    print("Rigid corr demo")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=.5)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=.5)

    theta = -58 * np.pi / 180
    orig_mat = np.array([[np.cos(theta), -np.sin(theta), -5],
                         [np.sin(theta), np.cos(theta), -1],
                         [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, orig_mat, img_1.shape[::-1])
    cv2.imwrite('imRigidB1.jpg', img_2)
    st = time.time()
    my_mat = findRigidCorr(img_1.astype(np.float), img_2.astype(np.float))
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print("my mat: \n", my_mat, "\n\n original mat: \n", orig_mat)
    my_rigid = cv2.warpPerspective(img_1, my_mat, img_1.shape[::-1])
    cv2.imwrite('imRigidB2.jpg', my_rigid)
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('CV Rigid')
    ax[0].imshow(img_2, cmap='gray')

    ax[1].set_title('My Rigid')
    ax[1].imshow(my_rigid, cmap='gray')

    plt.show()
    print("MSE =", MSE(my_rigid, img_2))


def imageWarpingDemo(img_path):
    print("---------------------------------------------------------------------------")
    print("Image Warping Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, 2],
                  [0, 1, 1],
                  [0, 0, 1]], dtype=np.float)

    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    im2 = warpImages(img_1.astype(np.float), img_2.astype(np.float), t)
    et = time.time()
    print("Time: {:.4f}".format(et - st))
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('My warping')
    ax[0].imshow(im2)

    ax[1].set_title('CV warping')
    ax[1].imshow(img_2)
    plt.show()




# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("---------------------------------------------------------------------------")
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("---------------------------------------------------------------------------")
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    print("Blending demo")
    print("---------------------------------------------------------------------------")
    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpeg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    # cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def MSE(a: np.ndarray, b: np.ndarray) -> float:
    return np.square(a - b).mean()


def main():
    print("ID:", myID())
    lkDemo('input/boxMan.jpg')
    hierarchicalkDemo('input/boxMan.jpg')
    compareLK('input/boxMan.jpg')
    translationlkdemo('input/myPic1.jpg')
    rigidlkdemo('input/myPic1.jpg')
    translationcorrdemo('input/myPic1.jpg')
    rigidcorrdemo('input/shapes.jpeg')
    imageWarpingDemo('input/sunset.jpg')
    pyrGaussianDemo('input/pyr_bit.jpg')
    pyrLaplacianDemo('input/pyr_bit.jpg')
    blendDemo()


if __name__ == '__main__':
    main()
