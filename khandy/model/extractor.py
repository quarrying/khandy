from abc import ABC, abstractmethod

import numpy as np

import khandy

__all__ = ['Extractor']


class Extractor(ABC):
    def __init__(self, feature_dim: int):
        self._feature_dim = feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    @abstractmethod
    def forward(self, image: khandy.KArray, **kwargs) -> khandy.KArray:
        pass

    def __call__(self, image: khandy.KArray, **kwargs) -> khandy.KArray:
        return self.forward(image, **kwargs)
    
    


