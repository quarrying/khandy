import cv2
import khandy
import numpy as np


def _convert_bool_matrix_to_int(bool_mat):
    hash_val = int(0)
    for item in bool_mat.flatten():
        hash_val <<= 1
        hash_val |= int(item)
    return hash_val
    
    
def calc_image_ahash(image):
    """Average Hashing

    References:
        http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
    """
    assert khandy.is_numpy_image(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (8, 8))
    
    mean_val = np.mean(resized)
    hash_mat = resized >= mean_val
    hash_val = _convert_bool_matrix_to_int(hash_mat)
    return f'{hash_val:016x}'
    
    
def calc_image_dhash(image):
    """Difference Hashing

    References:
        http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
    """
    assert khandy.is_numpy_image(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (9, 8))
    
    hash_mat = resized[:,:-1] >= resized[:,1:]
    hash_val = _convert_bool_matrix_to_int(hash_mat)
    return f'{hash_val:016x}'
    
    
def calc_image_phash(image):
    """Perceptual Hashing
    
    References:
        http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
    """
    assert khandy.is_numpy_image(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (32, 32))

    dct_coeff = cv2.dct(resized.astype(np.float32))
    reduced_dct_coeff = dct_coeff[:8, :8]

    # # mean of coefficients excluding the DC term (0th term)
    # mean_val = np.mean(reduced_dct_coeff.flatten()[1:])
    # median of coefficients
    median_val = np.median(reduced_dct_coeff)

    hash_mat = reduced_dct_coeff >= median_val
    hash_val = _convert_bool_matrix_to_int(hash_mat)
    return f'{hash_val:016x}'
    