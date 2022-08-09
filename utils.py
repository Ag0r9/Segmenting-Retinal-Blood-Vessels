import os
from typing import Tuple, Any, Dict

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix


def get_metrics(result: np.ndarray, real: np.ndarray) -> Tuple[float, float, float]:
    conf = confusion_matrix(real.flatten(), result.flatten())
    TN, FP, FN, TP = conf.ravel()

    accuracy = float(TP + TN) / float(TP + TN + FP + FN)
    sensitivity = TP / float(TP + FN)
    specificity = TN / float(TN + FP)

    return accuracy, sensitivity, specificity


def get_data(path: str = './data/HRF/') -> Dict[str, list[Any]]:
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
