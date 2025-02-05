from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import khandy
torch = khandy.import_torch()


__all__ = ['ClassifierResultItem', 'ClassifierResults', 'BaseTopKClassifier',
           'BaseClassifier', 'ClassCollectionItem', 'BaseCollectionsClassifier',
           'Gallery', 'find_topk', 'find_topk_in_gallery']


def find_topk(inputs, indices_list: Optional[List[List[int]]] = None, 
              k: int = 3, do_softmax: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if indices_list is not None:
        if do_softmax:
            valid_indices = list(set(sum(indices_list, [])))
            inputs = khandy.softmax(inputs, axis=-1, valid_indices=valid_indices)
        probs = khandy.sum_by_indices_list(inputs, indices_list, axis=-1)
    elif do_softmax:
        probs = khandy.softmax(inputs, axis=-1)
    else:
        probs = inputs
    topk_probs, topk_inds = khandy.top_k(probs, min(probs.shape[-1], k), axis=-1)
    return topk_probs, topk_inds


@dataclass
class ClassifierResultItem:
    conf: float
    class_index: int
    class_name: Optional[str] = None
    class_extra: Optional[Any] = None


class ClassifierResults(khandy.EqLenSequences):
    confs: khandy.KArray
    classes: khandy.KArray
    class_names: List[str]
    class_extras: List

    def __init__(self, confs: khandy.KArray, classes: Optional[khandy.KArray] = None, 
                 class_names: Optional[List[str]] = None, class_extras: Optional[List] = None):
        if confs.ndim == 2 and confs.shape[-1] == 1:
            confs = confs.flatten()
        if confs.ndim != 1:
            raise TypeError(f'Unsupported ndim for confs, got {confs.ndim}')
        
        if classes is None:
            if torch is not None and isinstance(confs, torch.Tensor):
                classes = torch.zeros((confs.shape[0],), dtype=torch.int32, device=confs.device)
            elif isinstance(confs, np.ndarray):
                classes = np.zeros((confs.shape[0],), dtype=np.int32)
            else:
                raise TypeError(f'Unsupported type for classes, got {type(classes)}')
        if classes.ndim == 2 and classes.shape[-1] == 1:
            classes = classes.flatten()
        if classes.ndim != 1:
            raise TypeError(f'Unsupported ndim for classes, got {classes.ndim}')
        
        assert class_names is None or khandy.is_list_of(class_names, str)
        assert class_extras is None or isinstance(class_extras, list)
        
        init_kwargs = {'confs': confs, 'classes': classes}
        if class_names is not None:
            init_kwargs['class_names'] = class_names
        if class_extras is not None:
            init_kwargs['class_extras'] = class_extras
        super().__init__(**init_kwargs)

    def __getitem__(self, key: Union[int, slice]) -> Union["ClassifierResults", ClassifierResultItem]:
        item = super().__getitem__(key)
        if type(key) == int:
            return ClassifierResultItem(
                conf=item.confs[0].item(),
                class_index=item.classes[0].item(),
                class_name=item.class_names[0] if 'class_names' in item else None,
                class_extra=item.class_extras[0] if 'class_extras' in item else None
            )
        return item
    
    
class BaseTopKClassifier(ABC):
    def __init__(
        self, 
        num_classes: int, 
        top_k: int = 3, 
        has_softmax: bool = False, 
        class_names: Optional[List[str]] = None,
        class_extras: Optional[List] = None
    ):
        self._num_classes = num_classes
        self._top_k = top_k
        self._has_softmax = has_softmax
        self._class_names = class_names
        self._class_extras = class_extras
        
    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def top_k(self) -> int:
        return self._top_k
    
    @top_k.setter
    def top_k(self, value: int):
        assert isinstance(value, int)
        if value <= 0:
            value = self.num_classes
        if value > self.num_classes:
            value = self.num_classes
        self._top_k = value
        
    @property
    def has_softmax(self) -> bool:
        return self._has_softmax

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._class_names
    
    @class_names.setter
    def class_names(self, value: Optional[List[str]]):
        if value is not None:
            assert khandy.is_list_of(value, str) and len(value) == self.num_classes
        self._class_names = value
        
    @property
    def class_extras(self) -> Optional[List]:
        return self._class_extras
    
    @class_extras.setter
    def class_extras(self, value: Optional[List]):
        if value is not None:
            assert isinstance(value, list) and len(value) == self.num_classes
        self._class_extras = value
        
    @abstractmethod
    def forward(self, image: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def __call__(self, image: np.ndarray, **kwargs) -> ClassifierResults:
        outputs = self.forward(image, **kwargs)
        topk_probs, topk_inds = find_topk(outputs, None, self.top_k, not self.has_softmax)
        results = ClassifierResults(confs=topk_probs[0], classes=topk_inds[0])
        if self.class_names is not None:
            results.class_names = [self.class_names[ind.item()] for ind in results.classes]
        if self.class_extras is not None:
            results.class_extras = [self.class_extras[ind.item()] for ind in results.classes]
        return results
    
    
class BaseClassifier(BaseTopKClassifier):
    def __init__(
        self, 
        num_classes: int, 
        has_softmax: bool = False, 
        class_names: Optional[List[str]] = None,
        class_extras: Optional[List] = None
    ):
        super().__init__(num_classes, 1, has_softmax, class_names, class_extras)

    def __call__(self, image: np.ndarray, **kwargs) -> ClassifierResultItem:
        return super().__call__(image, **kwargs)[0]


@dataclass
class ClassCollectionItem:
    indices: List[int]
    name: Optional[str] = None
    extra: Optional[Any] = None


def find_topk_in_collections(
        inputs, collections: Dict[str, List[ClassCollectionItem]], 
        k: int = 3, do_softmax: bool = False
    ) -> List[Dict[str, ClassifierResults]]:
    num_examples = len(inputs)
    results = [{} for _ in range(num_examples)]
    for collection_name, collection_val in collections.items(): # #collection
        indices_list = [item.indices for item in collection_val]
        topk_probs, topk_inds = find_topk(inputs, indices_list, k, do_softmax)
        for example_ind in range(num_examples): # batch_size
            one_results = ClassifierResults(confs=topk_probs[example_ind], classes=topk_inds[example_ind])
            class_names = []
            for ind in one_results.classes:
                class_name = collection_val[ind.item()].name
                if class_name is None:
                    class_name = f'{collection_name}#unnamed_class#{ind.item()}'
                class_names.append(class_name)
            one_results.class_names = class_names
            one_results.class_extras = [collection_val[ind.item()].extra for ind in one_results.classes]
            results[example_ind][collection_name] = one_results
    # (batch_size, #collection, k)
    return results


class BaseCollectionsClassifier(BaseTopKClassifier):
    def __init__(self, num_classes: int, 
                 collections: Dict[str, List[ClassCollectionItem]], 
                 top_k: int, has_softmax=False):
        super().__init__(num_classes, top_k, has_softmax)
        self._collections = collections

    @property
    def collections(self) -> Dict[str, List[ClassCollectionItem]]:
        return self._collections
    
    def __call__(self, image: np.ndarray, **kwargs) -> Dict[str, ClassifierResults]:
        outputs = self.forward(image, **kwargs)
        topk_results = find_topk_in_collections(outputs, self.collections, self.top_k, not self.has_softmax)
        return topk_results[0]


class Gallery:
    def __init__(self, features: np.ndarray, 
                 collections: Dict[str, List[ClassCollectionItem]], 
                 model_name: str):
        self._features = features
        self._collections = collections
        self._model_name = model_name
    
    @property
    def features(self) -> np.ndarray:
        return self._features
    
    @property
    def collections(self) -> Dict[str, List[ClassCollectionItem]]:
        return self._collections
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def size(self) -> int:
        return len(self.features)
    
    @property
    def feature_dim(self) -> int:
        return self.features.shape[-1]


def find_topk_in_gallery(inputs, gallery: Gallery, k: int = 3) -> List[Dict[str, ClassifierResults]]:
    logits = np.dot(inputs, gallery.features.T)
    return find_topk_in_collections(logits, gallery.collections, k, do_softmax=True)

