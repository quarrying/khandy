from typing import Optional

import numpy as np

from .detector import *
from .extractor import *

import khandy

__all__ = ['extract_feature']


def extract_feature(image: np.ndarray, extractor: Extractor, 
                    det_objects: Optional[DetObjects] = None,
                    detector: Optional[BaseDetector] = None,
                    **detector_kwargs) -> DetObjects:
    if det_objects is not None:
        pass
    elif detector is not None:
        det_objects = detector(image, **detector_kwargs)
    else:
        det_objects.feats = np.expand_dims(extractor(image), axis=0)
        return det_objects
    
    feats = np.empty((len(det_objects), extractor.feature_dim), dtype=np.float32)
    for det_obj in det_objects:
        cropped = khandy.crop_image(
            image, round(det_obj.x_min), round(det_obj.y_min),
            round(det_obj.x_max), round(det_obj.y_max))
        feats.append(extractor(cropped))
    det_objects.feats = feats
    return det_objects

