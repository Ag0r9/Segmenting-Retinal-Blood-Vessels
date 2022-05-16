import os
from typing import Tuple

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix


def get_metrics(result: np.ndarray, real: np.ndarray):
    conf = confusion_matrix(result.flatten(), real.flatten())
    TN, FP, FN, TP = conf.ravel()

    accuracy = 1.0 * (TP + TN) / (TP + TN + FP + FN)
    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (TN + FP)

    return accuracy, sensitivity, specificity


def get_data(path='./data/HRF/'):
    files = sorted(os.listdir(path))
    data = {
        'original': [],
        'labeled': [],
        'mask': []
    }
    for file in files:
        if file.endswith('h.jpg'):
            data['original'].append(cv2.imread(f'{path}{file}'))
        elif file.endswith('h.tif'):
            data['labeled'].append(cv2.imread(f'{path}{file}', 0))
        elif file.endswith('h_mask.tif'):
            data['mask'].append(cv2.imread(f'{path}{file}', 0))
    return data