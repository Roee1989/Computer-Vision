# imports for hw1 (you can add any other library as well)
import cv2
from scipy import io as sio
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt


def createGaussianPyramid(im, sigma0, k, levels):
    GaussianPyramid = []
    for i in range(len(levels)):
        sigma_ = sigma0 * k ** levels[i]
        size = int(np.floor( 3 * sigma_ * 2) + 1)
        blur = cv2.GaussianBlur(im,(size,size),sigma_)
        GaussianPyramid.append(blur)
    return np.stack(GaussianPyramid)


def displayPyramid(pyramid):
    plt.figure(figsize=(16,5))
    plt.imshow(np.hstack(pyramid), cmap='gray')
    plt.axis('off')
    plt.show()


def createDoGPyramid(GaussianPyramid, levels):
    # Produces DoG Pyramid
    # inputs
    # Gaussian Pyramid - A matrix of grayscale images of size
    #                    (len(levels), shape(im))
    # levels      - the levels of the pyramid where the blur at each level is
    #               outputs
    # DoG Pyramid - size (len(levels) - 1, shape(im)) matrix of the DoG pyramid
    #               created by differencing the Gaussian Pyramid input
    L = GaussianPyramid.shape[0]
    Imx = GaussianPyramid.shape[1]
    Imy = GaussianPyramid.shape[2]
    DoGPyramid = np.empty([L - 1, Imx, Imy])
    for i in range(L - 1):
        DoGPyramid[i] = GaussianPyramid[i+1] - GaussianPyramid[i]
    DoGLevels = levels
    return DoGPyramid, DoGLevels


def computePrincipalCurvature(DoGPyramid):
    # Edge Suppression
    #  Takes in DoGPyramid generated in createDoGPyramid and returns
    #  PrincipalCurvature,a matrix of the same size where each point contains the
    #  curvature ratio R for the corre-sponding point in the DoG pyramid
    #
    #  INPUTS
    #  DoG Pyramid - size (len(levels) - 1, shape(im)) matrix of the DoG pyramid
    #
    #  OUTPUTS
    #  PrincipalCurvature - size (len(levels) - 1, shape(im)) matrix where each
    #                       point contains the curvature ratio R for the
    #                       corresponding point in the DoG pyramid
    kers=3; nlevels = DoGPyramid.shape[0]
    Dyy = np.empty(DoGPyramid.shape)
    Dxx = np.empty(DoGPyramid.shape)
    Dyx = np.empty(DoGPyramid.shape)
    for i in range(nlevels):
        Dyy[i] = cv2.Sobel(DoGPyramid[i], cv2.CV_64FC1, 2, 0, ksize=kers)
        Dxx[i] = cv2.Sobel(DoGPyramid[i], cv2.CV_64FC1, 0, 2, ksize=kers)
        Dyx[i] = cv2.Sobel(DoGPyramid[i], cv2.CV_64FC1, 1, 1, ksize=kers)
    PrincipalCurvature = np.square(Dxx + Dyy)/(Dxx * Dyy - Dyx * Dyx)
    return PrincipalCurvature


def getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature,
                    th_contrast, th_r):
    #     Returns local extrema points in both scale and space using the DoGPyramid
    #     INPUTS
    #         DoG_pyramid - size (len(levels) - 1, imH, imW ) matrix of the DoG pyramid
    #         DoG_levels  - The levels of the pyramid where the blur at each level is
    #                       outputs
    #         principal_curvature - size (len(levels) - 1, imH, imW) matrix contains the
    #                       curvature ratio R
    #         th_contrast - remove any point that is a local extremum but does not have a
    #                       DoG response magnitude above this threshold
    #         th_r        - remove any edge-like points that have too large a principal
    #                       curvature ratio
    #      OUTPUTS
    #         locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
    #                scale and space, and also satisfies the two thresholds.
    DoGPyramid_padded = np.pad(DoGPyramid, 1)
    locsDoG = []
    for k in range(1, DoGPyramid.shape[0]+1):
        for j in range(1, DoGPyramid.shape[2]+1):
            for i in range(1, DoGPyramid.shape[1]+1):
                patch = np.concatenate((DoGPyramid_padded[k, i-1:i+2, j-1:j+2].flatten(),
                                        [DoGPyramid_padded[k + 1, i, j]], [DoGPyramid_padded[k - 1, i, j]]))
                if np.argmax(patch) == 4 and patch[4] > th_contrast and PrincipalCurvature[k-1, i-1, j-1] > th_r:
                    locsDoG.append([j-1, i-1, k-1])
    return locsDoG


def DoGdetector(im, sigma0, k, levels,
    th_contrast, th_r):
    #     Putting it all together
    #     Inputs          Description
    #     --------------------------------------------------------------------------
    #     im              Grayscale image with range [0,1].
    #     sigma0          Scale of the 0th image pyramid.
    #     k               Pyramid Factor.  Suggest sqrt(2).
    #     levels          Levels of pyramid to construct. Suggest -1:4.
    #     th_contrast     DoG contrast threshold.  Suggest 0.03.
    #     th_r            Principal Ratio threshold.  Suggest 12.
    #     Outputs         Description
    #     --------------------------------------------------------------------------
    #     locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
    #                     in both scale and space, and satisfies the two thresholds.
    #     gauss_pyramid   A matrix of grayscale images of size (len(levels),imH,imW)
    GaussianPyramid = createGaussianPyramid(im, sigma0, k, levels)
    [DoGPyramid, DoGLevels] = createDoGPyramid(GaussianPyramid, levels)
    PrincipalCurvature = computePrincipalCurvature(DoGPyramid)
    locsDoG = getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature, th_contrast, th_r)
    return locsDoG, GaussianPyramid




def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    plt.show()


def briefMatch(desc1, desc2, ratio):
    #     performs the descriptor matching
    #     inputs  : desc1 , desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
    #                                 n is the number of bits in the brief
    #     outputs : matches - p x 2 matrix. where the first column are indices
    #                                         into desc1 and the second column are indices into desc2
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]
    matches = np.stack((ix1,ix2), axis=-1)
    return matches


def makeTestPattern(patchWidth, nbits):
    compareX = np.random.randint(0, patchWidth**2, size=nbits)
    compareY = np.random.randint(0, patchWidth**2, size=nbits)
    sio.savemat("testPattern.mat", {'compareX': compareX, 'compareY': compareY})
    return compareX, compareY


def computeBrief(im, GaussianPyramid, locsDoG, k, levels,
compareX, compareY):
    Imx = im.shape[1]; Imy = im.shape[0]
    desc = []
    locs = [x for x in locsDoG if (Imy - 4 > x[1] > 3 and Imx - 4 > x[0] > 3)]
    for loc in locs:
        patch = im[loc[1] - 4:loc[1] + 5, loc[0] - 4:loc[0] + 5].flatten()
        desc.append((patch[compareX] < patch[compareY]).astype(int)[0])
    return np.asarray(locs), desc


def briefLite(im):
    sigma0 = 1; k = np.sqrt(2); levels = [-1, 0, 1, 2, 3, 4]; th_contrast = 0.03; th_r = 12
    im_grayscale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
    contents = sio.loadmat("testPattern.mat")
    compareX = contents['compareX']
    compareY = contents['compareY']
    [locsDoG, GaussianPyramid] = DoGdetector(im_grayscale, sigma0, k, levels, th_contrast, th_r)
    [locs, desc] = computeBrief(im_grayscale, GaussianPyramid, locsDoG, k, levels,compareX, compareY)
    return locs, desc




sigma0 = 1; k = np.sqrt(2); levels = [-1, 0, 1, 2, 3, 4]; th_contrast = 0.03; th_r = 12
# 1.1
im = cv2.imread('data/model_chickenbroth.jpg')
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
_ = plt.axis('off')
plt.show()

# 1.2
im_grayscale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
pyramid = createGaussianPyramid(im_grayscale, sigma0, k, levels)
displayPyramid(pyramid)

# 1.3
[DoGPyramid, DoGLevels] = createDoGPyramid(pyramid, levels)

# 1.4
PrincipalCurvature = computePrincipalCurvature(DoGPyramid)

# 1.5

locsDoG = getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature, th_contrast, th_r)

# 1.6
[locsDoG, GaussianPyramid] = DoGdetector(im_grayscale, sigma0, k, levels, th_contrast, th_r)
corners = np.zeros(im_grayscale.shape)
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
_ = plt.axis('off')
for corner in locsDoG:
    plt.scatter(corner[0], corner[1], s=5, c='red', marker='o')
plt.show()


# 2.4
patchWidth = 9; nbits = 256; ratio = 0.8
[compareX, compareY] = makeTestPattern(patchWidth, nbits)
im1 = cv2.imread('data/pf_scan_scaled.jpg')
im2 = cv2.imread('data/pf_floor.jpg')
locs1, desc1 = briefLite(im1)
locs2, desc2 = briefLite(im2)
matches = briefMatch(desc1, desc2, ratio)
plotMatches(im1, im2, matches, locs1, locs2)


# 2.5
patchWidth = 9; nbits = 256; ratio = 0.8
[compareX, compareY] = makeTestPattern(patchWidth, nbits)
im = cv2.imread('data/model_chickenbroth.jpg')
im_rot = cv2.imread('data/model_chickenbroth.jpg')
locs1, desc1 = briefLite(im)

rot_angles = np.linspace(0, 90, 10)
matches = []
for rot_angle in rot_angles:
    rows,cols,dim = im_rot.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rot_angle,1)
    for i in range(dim):
        im_rot[:,:,i] = cv2.warpAffine(im_rot[:,:,i],M,(cols,rows))
    locs2, desc2 = briefLite(im_rot)
    matches.append(len(briefMatch(desc1, desc2, ratio)))
    # plotMatches(im, im_rot, matches, locs1, locs2)

plt.bar(np.arange(len(rot_angles)), matches, align='center')
plt.xticks(np.arange(len(rot_angles)), rot_angles)
plt.ylabel('number of matches')
plt.title('rotation angle[deg]')

plt.show()
