import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform, AffineTransform
from skimage.transform import warp
from skimage.filters import gaussian
from numpy.linalg import inv

DEFAULT_TRANSFORM = ProjectiveTransform


def find_keypoints(img, n_keypoints=200):

    img_gray = rgb2gray(img)

    descriptor_extractor = ORB(n_keypoints=n_keypoints)
    descriptor_extractor.detect_and_extract(img_gray)

    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors

    return keypoints, descriptors


def center_and_normalize_points(points):


    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    matrix = np.zeros((3, 3))

    n = points.shape[0]

    C_x = np.mean(points[:, 0])
    C_y = np.mean(points[:, 1])

    N = np.sqrt(2) / np.sqrt(np.sum((points - [C_x, C_y]) ** 2) / n)

    M = np.array([
        [N, 0, -N * C_x],
        [0, N, -N * C_y],
        [0, 0, 1]])

    pointsh_c = (M @ pointsh).T

    points_c = pointsh_c[:, :2] / pointsh_c[:, 2:]

    return M, points_c


def find_homography(src_keypoints, dest_keypoints):


    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)

    H = np.zeros((3, 3))


    n = src_keypoints.shape[0]

    A = np.zeros((2 * n, 9))
    for i in range(n):
        A[2 * i] = [-src[i][0], -src[i][1], -1, 0, 0, 0, 
                    dest[i][0] * src[i][0], dest[i][0] * src[i][1], dest[i][0]]
        A[2 * i + 1] = (0, 0, 0, -src[i][0], -src[i][1], -1,
                        dest[i][1] * src[i][0], dest[i][1] * src[i][1], dest[i][1])

    _, _, Vh = np.linalg.svd(A)

    H = np.linalg.inv(dest_matrix) @ Vh[-1].reshape(3, 3) @ src_matrix

    return H

def ransac_tr(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials=5000, residual_threshold=3, return_matches=False):


    matches = match_descriptors(src_descriptors, dest_descriptors)

    src_keypoints = src_keypoints[matches[:, 0]]
    dest_keypoints = dest_keypoints[matches[:, 1]]

    n = src_keypoints.shape[0]

    src_keypoints_h = np.row_stack([src_keypoints.T, np.ones(n, )])

    best_inliers = np.array([])

    for _ in range(max_trials):

        cur_ind = np.random.choice(np.arange(n), 4, replace=False)

        cur_H = find_homography(src_keypoints[cur_ind], dest_keypoints[cur_ind])

        cur_dest_h = (cur_H @ src_keypoints_h).T

        cur_dest = cur_dest_h[:, :2] / cur_dest_h[:, 2:]

        cur_inliers = np.where(
            np.sqrt(np.sum((cur_dest - dest_keypoints) ** 2, axis=1)) < residual_threshold)[0]

        if len(cur_inliers) > len(best_inliers):
            best_inliers = cur_inliers

    best_H = find_homography(src_keypoints[best_inliers], 
                             dest_keypoints[best_inliers])

    if return_matches:
        return ProjectiveTransform(best_H), matches[best_inliers]

    return ProjectiveTransform(best_H)
    

def find_center_warps(forward_transforms):

    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2

    result = [None] * image_count
    result[center_index] = DEFAULT_TRANSFORM()


    for i in range(image_count):
        if i < center_index:
            cur_trans = forward_transforms[i]
            for j in range(i + 1, center_index):
                cur_trans += forward_transforms[j]
            result[i] = ProjectiveTransform(
                inv(rotate_transform_matrix(cur_trans)))
        elif i > center_index:
            cur_trans = forward_transforms[center_index]
            for j in range(center_index + 1, i):
                cur_trans += forward_transforms[j]
            result[i] = rotate_transform_matrix(cur_trans)

    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [0, height],
                            [width, height],
                            [width, 0]])

        yield transform.inverse(corners)[:, ::-1]


def get_min_max_coords(corners):
    #Get minimum and maximum coordinates of corners
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):

    corners = get_corners(image_collection, simple_center_warps)
    corners = np.array(list(corners))
    cords = get_min_max_coords(corners)

    output_shape = np.ceil(cords[1] - cords[0]).astype(int)

    AT = AffineTransform(translation=cords[0][::-1])
    final_center_warps = [AT + PT for PT in simple_center_warps]

    return final_center_warps, output_shape


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):

    mask = np.ones(image.shape[:2], dtype=bool)
    img_wrap = warp(image, transform, output_shape=output_shape)
    mask_wrap = warp(mask, transform, output_shape=output_shape)

    return img_wrap, mask_wrap


def merge_final_pano(image_collection, final_center_warps, output_shape):

    result = np.zeros(np.append(output_shape, 3))
    result_mask = np.zeros(output_shape, dtype=np.bool8)

    for img, T in zip(image_collection, final_center_warps):
        img_wrap, mask_wrap = warp_image(img, T, output_shape)
        result[mask_wrap] = img_wrap[mask_wrap]

    return result


def get_gaussian_pyramid(image, layers=4, sigma=10):

    if image.dtype != 'float':
        image = image / 255

    G = [None] * layers
    G[0] = image
    
    for i in range(1, layers):
        G[i] = gaussian(G[i - 1], sigma=sigma)

    return tuple(G)
    

def get_laplacian_pyramid(image, layers=4, sigma=10):


    if image.dtype != 'float':
        image = image / 255
    
    G = get_gaussian_pyramid(image, layers, sigma)
    L = [None] * len(G)
    L[-1] = G[-1]

    for i in range(len(G) - 1):
        L[i] = G[i] - G[i + 1]

    return tuple(L)







def gaussian_merging_pano(image_collection, final_center_warps, output_shape, n_layers=3, image_sigma=1, merge_sigma=10):

    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype='float64')

    mask_wraps = []
    img_wraps = []
    for img, T in zip(image_collection, final_center_warps):
        img_wrap, mask_wrap = warp_image(img, T, output_shape)
        img_wraps.append(img_wrap)
        mask_wraps.append(mask_wrap)

    # Split images with no intersection
    for i in range(len(mask_wraps) - 1):

        # Find intersections
        inter = np.add(mask_wraps[i], mask_wraps[i + 1], dtype='int') == 2

        # Find intersection center
        v_sum = np.sum(inter, axis=0)
        inter_start = np.where(v_sum > 0)[0][0]
        inter_end = np.where(v_sum > 0)[0][-1]
        line = (inter_end + inter_start) // 2

        inter_left = inter.copy()
        inter_left[:, line:] = False 
        
        inter_right = inter.copy()
        inter_right[:, :line] = False 

        mask_wraps[i][inter_right] = False

        mask_wraps[i + 1][inter_left] = False

        for j in range(i + 1, len(mask_wraps)):
            mask_wraps[j][mask_wraps[i]] = False

    mask_wraps_G = np.array([get_gaussian_pyramid(
        mask_wrap, n_layers, merge_sigma) for mask_wrap in mask_wraps])

    # Normalize
    norm = np.sum(mask_wraps_G, axis=0)
    mask_wraps_G = np.divide(mask_wraps_G, norm, where=norm!=0)

    img_wraps_L = np.array([get_laplacian_pyramid(
        img_wrap, n_layers, image_sigma) for img_wrap in img_wraps])

    result = np.sum(mask_wraps_G[..., np.newaxis] * img_wraps_L, axis=(0, 1))

    return result.clip(0, 1)


