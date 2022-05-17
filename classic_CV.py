import cv2
import numpy as np
from skimage.filters.ridges import frangi


def normalize(image):
    filtered_min, filtered_max = image.min(), image.max()
    return (1 - (image - filtered_min) / (filtered_max - filtered_min)) * 255


def create_result_classic(image, mask):
    filtered = frangi(image)
    mask = (frangi(mask))
    mask = cv2.blur(mask, (30, 30))
    mask = normalize(mask)
    _, mask = cv2.threshold(mask, 190, 255, cv2.THRESH_BINARY)
    image = normalize(filtered)
    _, image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)
    image, mask = image.astype(np.uint8), mask.astype(np.uint8)
    image = cv2.bitwise_and(image, image, mask=mask)
    return image
