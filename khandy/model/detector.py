import itertools
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Mapping, Optional, Tuple, Union

import numpy as np

import khandy

torch = khandy.import_torch()

__all__ = ['DetObjectItem', 'DetObjectSortDir', 'DetObjectSortBy', 'DetObjects', 
           'BaseDetector', 'Index2LabelType', 'label_image_by_detector',
           'concat_det_objects', 'detect_in_det_objects']


@dataclass
class DetObjectItem:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    conf: float
    class_index: int
    class_name: str
    
    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


class DetObjectSortDir(Enum):
    ASC = auto()
    DESC = auto()


class DetObjectSortBy(Enum):
    BY_AREA = auto()
    BY_CONF = auto()
    BY_CLASS = auto()


class DetObjects(khandy.EqLenSequences):
    boxes: khandy.KArray
    confs: khandy.KArray
    classes: khandy.KArray
    class_names: List[str]

    def __init__(
        self,
        boxes: Optional[khandy.KArray] = None,
        confs: Optional[khandy.KArray] = None,
        classes: Optional[khandy.KArray] = None,
        class_names: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(
            boxes=boxes, 
            confs=confs, 
            classes=classes, 
            class_names=class_names, 
            **kwargs
        )
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'boxes':
            value = self._setup_boxes(value)
        elif name == 'confs':
            value = self._setup_confs(value)
        elif name == 'classes':
            value = self._setup_classes(value)
        elif name == 'class_names':
            value = self._setup_class_names(value)
        super().__setattr__(name, value)
        
    def _setup_boxes(self, boxes: Optional[khandy.KArray] = None) -> khandy.KArray:
        if boxes is None:
            boxes = np.empty((len(self), 4), dtype=np.float32)
        if torch is not None and isinstance(boxes, torch.Tensor):
            pass
        elif isinstance(boxes, np.ndarray):
            pass
        else:
            raise TypeError(f'Unsupported type for boxes, got {type(boxes)}')
            
        assert boxes.ndim == 2, f'boxes ndim is not 2, got {boxes.ndim}'
        assert boxes.shape[1] == 4, f'boxes last axis size is not 4, got {boxes.shape[1]}'
        return boxes
 
    def _setup_confs(self, confs: Optional[khandy.KArray] = None) -> khandy.KArray:
        if confs is None:
            if torch is not None and isinstance(self.boxes, torch.Tensor):
                confs = torch.ones((len(self), 1), dtype=torch.float32, device=self.boxes.device)
            elif isinstance(self.boxes, np.ndarray):
                confs = np.ones((len(self), 1), dtype=np.float32)
            else:
                raise TypeError(f'Unsupported type for confs, got {type(confs)}')
        if confs.ndim == 1:
            confs = confs.reshape((-1, 1))
            
        assert confs.ndim == 2, f'confs ndim is not 2, got {confs.ndim}'
        assert confs.shape[1] == 1, f'confs last axis size is not 1, got {confs.shape[1]}'
        return confs

    def _setup_classes(self, classes: Optional[khandy.KArray] = None) -> khandy.KArray:
        if classes is None:
            if torch is not None and isinstance(self.boxes, torch.Tensor):
                classes = torch.zeros((len(self), 1), dtype=torch.int32, device=self.boxes.device)
            elif isinstance(self.boxes, np.ndarray):
                classes = np.zeros((len(self), 1), dtype=np.int32)
            else:
                raise TypeError(f'Unsupported type for classes, got {type(classes)}')
        if classes.ndim == 1:
            classes = classes.reshape((-1, 1))
        
        assert classes.ndim == 2, f'classes ndim is not 2, got {classes.ndim}'
        assert classes.shape[1] == 1, f'classes last axis size is not 1, got {classes.shape[1]}'
        return classes

    def _setup_class_names(self, class_names: Optional[List[str]] = None) -> List[str]:
        if class_names is None:
            class_names = [f'unnamed_class#{class_ind}' for class_ind in self.classes.flatten()]
            
        assert khandy.is_seq_of(class_names, str), f'class_names must be list of str'
        return class_names
    
    def __getitem__(self, key: Union[int, slice]) -> Union["DetObjects", DetObjectItem]:
        item = super().__getitem__(key)
        if type(key) == int:
            return DetObjectItem(
                x_min=item.boxes[0, 0],
                y_min=item.boxes[0, 1],
                x_max=item.boxes[0, 2],
                y_max=item.boxes[0, 3],
                conf=item.confs[0].item(),
                class_index=item.classes[0].item(),
                class_name=item.class_names[0]
            )
        return item
    
    def filter_by_class_index(self, interested_class_inds, inplace=False) -> "DetObjects":
        warnings.warn(
            "filter_by_class_index is deprecated, use filter_by_class_indices instead.",
            DeprecationWarning
        )
        assert isinstance(self.classes, np.ndarray)
        mask = np.zeros((len(self.classes),), dtype=bool)
        for class_ind in interested_class_inds:
            mask = np.logical_or(mask, self.classes[:, 0] == class_ind)
        keep = np.nonzero(mask)[0]
        return self.filter(keep, inplace)

    def filter_by_class_indices(
        self,
        interested: Optional[Union[Tuple[int], List[int]]] = None,
        ignored: Optional[Union[Tuple[int], List[int]]] = None,
        inplace: bool = False
    ) -> "DetObjects":
        if interested is not None and ignored is not None:
            raise ValueError("You cannot specify both 'interested' and 'ignored' at the same time.")
        if interested is None and ignored is None:
            raise ValueError("You must specify either 'interested' or 'ignored'.")
        
        mask = np.empty((len(self.classes),), dtype=bool)
        for k, class_ind in enumerate(self.classes):
            if ignored is not None:
                mask[k] = class_ind not in ignored
            elif interested is not None:
                mask[k] = class_ind in interested

        keep = np.nonzero(mask)[0]
        return self.filter(keep, inplace)
    
    def filter_by_class_names(
        self, 
        interested: Optional[Union[Tuple[str], List[str]]] = None,
        ignored: Optional[Union[Tuple[str], List[str]]] = None, 
        inplace: bool = False
    ) -> "DetObjects":
        if interested is not None and ignored is not None:
            raise ValueError("You cannot specify both 'interested' and 'ignored' at the same time.")
        if interested is None and ignored is None:
            raise ValueError("You must specify either 'interested' or 'ignored'.")
        
        mask = np.empty((len(self.class_names),), dtype=bool)
        for k, class_name in enumerate(self.class_names):
            if ignored is not None:
                mask[k] = class_name not in ignored
            elif interested is not None:
                mask[k] = (class_name in interested)

        keep = np.nonzero(mask)[0]
        return self.filter(keep, inplace)

    def filter_by_min_area(self, min_area, inplace=False) -> "DetObjects":
        assert isinstance(self.boxes, np.ndarray)
        widths = self.boxes[:, 2] - self.boxes[:, 0]
        heights = self.boxes[:, 3] - self.boxes[:, 1] 
        mask = widths * heights >= min_area
        keep = np.nonzero(mask)[0]
        return self.filter(keep, inplace)

    def filter_by_min_size(self, min_width, min_height, inplace=False) -> "DetObjects":
        assert isinstance(self.boxes, np.ndarray)
        keep = khandy.filter_small_boxes(self.boxes, min_width, min_height)
        return self.filter(keep, inplace)

    def filter_by_conf(self, conf_thresh, inplace=False):
        assert isinstance(self.confs, np.ndarray)
        if isinstance(conf_thresh, (list, np.ndarray)):
            conf_thresh = np.array(conf_thresh)
            mask = self.confs > conf_thresh[self.classes]
        else:
            mask = self.confs > conf_thresh
        keep = np.nonzero(mask)[0]
        return self.filter(keep, inplace)

    def nms(self, iou_thresh, ratio_type='iou', inplace=False) -> "DetObjects":
        assert isinstance(self.confs, np.ndarray)
        keep = khandy.non_max_suppression(self.boxes, self.confs, iou_thresh, self.classes, ratio_type)
        return self.filter(keep, inplace)

    def sort(self, sort_by: DetObjectSortBy, direction:  DetObjectSortDir = DetObjectSortDir.DESC) -> "DetObjects":
        assert isinstance(self.confs, np.ndarray)
        if sort_by == DetObjectSortBy.BY_CONF:
            sorted_inds = np.argsort(self.confs, axis=0)
        elif sort_by == DetObjectSortBy.BY_AREA:
            boxes = khandy.Boxes(self.boxes)
            sorted_inds = np.argsort(boxes.areas, axis=0)
        elif sort_by == DetObjectSortBy.BY_CLASS:
            sorted_inds = np.argsort(self.classes, axis=0)
        if direction == DetObjectSortDir.DESC:
            sorted_inds = sorted_inds[::-1]
        return self[sorted_inds]


class BaseDetector(ABC):
    def __init__(
        self, 
        num_classes: Optional[int] = None,
        conf_thresh: Optional[Union[float, List[float], Tuple[float], np.ndarray]] = None,
        iou_thresh: Optional[float] = None,
        min_width: Optional[Union[int, float]] = None,
        min_height: Optional[Union[int, float]] = None,
        min_area: Optional[Union[int, float]] = None,
        class_names: Optional[Union[List[str], Tuple[str]]] = None,
        sort_by: Optional[DetObjectSortBy] = None,
        sort_dir: Optional[DetObjectSortDir] = DetObjectSortDir.DESC
    ):
        self._num_classes = num_classes
        self._conf_thresh = conf_thresh
        self._iou_thresh = iou_thresh
        self._min_width = min_width
        self._min_height = min_height
        self._min_area = min_area
        self._class_names = class_names
        self._sort_by = sort_by
        self._sort_dir = sort_dir

    @property
    def num_classes(self) -> Optional[int]:
        return self._num_classes
    
    @property
    def conf_thresh(self) -> Optional[Union[float, np.ndarray]]:
        return self._conf_thresh
    
    @conf_thresh.setter
    def conf_thresh(self, value: Optional[Union[float, List[float], Tuple[float], np.ndarray]]):
        if value is None or isinstance(value, float):
            pass
        elif isinstance(value, (list, tuple)):
            assert khandy.is_seq_of(value, float) and len(value) == self.num_classes
            value = np.array(value)
        elif isinstance(value, np.ndarray):
            assert value.shape == (self.num_classes,) or value.shape == (self.num_classes, 1)
            value = value.flatten()
        else:
            raise TypeError(f'Unsupported type for conf_thresh, got {type(value)}')
        self._conf_thresh = value
    
    @property
    def iou_thresh(self) -> Optional[float]:
        return self._iou_thresh
    
    @iou_thresh.setter
    def iou_thresh(self, value: Optional[float]):
        assert value is None or isinstance(value, float), f'Unsupported type for iou_thresh, got {type(value)}'
        assert value is None or (0 < value < 1), f'iou_thresh must be in (0, 1), got {value}'
        self._iou_thresh = value
        
    @property
    def min_width(self) -> Optional[Union[int, float]]:
        return self._min_width
    
    @min_width.setter
    def min_width(self, value: Optional[Union[int, float]]):
        assert value is None or isinstance(value, (int, float)), f'Unsupported type for min_width, got {type(value)}'
        assert value is None or value >= 0, f'min_width must be >= 0, got {value}'
        self._min_width = value
        
    @property
    def min_height(self) -> Optional[Union[int, float]]:
        return self._min_height
    
    @min_height.setter
    def min_height(self, value: Optional[Union[int, float]]):
        assert value is None or isinstance(value, (int, float)), f'Unsupported type for min_height, got {type(value)}'
        assert value is None or value >= 0, f'min_height must be >= 0, got {value}'
        self._min_height = value
        
    @property
    def min_area(self) -> Optional[Union[int, float]]:
        return self._min_area
    
    @min_area.setter
    def min_area(self, value: Optional[Union[int, float]]):
        assert value is None or isinstance(value, (int, float)), f'Unsupported type for min_area, got {type(value)}'
        assert value is None or value >= 0, f'min_area must be >= 0, got {value}'
        self._min_area = value
        
    @property
    def class_names(self) -> Optional[List[str]]:
        return self._class_names
    
    @class_names.setter
    def class_names(self, value: Optional[Union[List[str], Tuple[str]]]):
        if value is not None:
            assert khandy.is_seq_of(value, str) and len(value) == self.num_classes
            value = list(value)
        self._class_names = value
        
    @property
    def sort_by(self) -> Optional[DetObjectSortBy]:
        return self._sort_by
    
    @sort_by.setter
    def sort_by(self, value: Optional[DetObjectSortBy]):
        self._sort_by = value
        
    @property
    def sort_dir(self) -> DetObjectSortDir:
        return self._sort_dir
    
    @sort_dir.setter
    def sort_dir(self, value: Optional[DetObjectSortDir]):
        self._sort_dir = value or DetObjectSortDir.DESC

    @abstractmethod
    def forward(self, image: khandy.KArray, **kwargs) -> DetObjects:
        pass
    
    def filter_by_conf(self, det_objects: DetObjects) -> DetObjects:
        if self.conf_thresh is not None:
            return det_objects.filter_by_conf(self.conf_thresh, inplace=True)
        return det_objects
    
    def filter_by_min_size(self, det_objects: DetObjects) -> DetObjects:
        if self.min_width is not None or self.min_height is not None:
            return det_objects.filter_by_min_size(self.min_width, self.min_height, inplace=True)
        return det_objects
    
    def filter_by_min_area(self, det_objects: DetObjects) -> DetObjects:
        if self.min_area is not None:
            return det_objects.filter_by_min_area(self.min_area, inplace=True)
        return det_objects
    
    def nms(self, det_objects: DetObjects, ratio_type: str = 'iou') -> DetObjects:
        if self.iou_thresh is not None:
            return det_objects.nms(self.iou_thresh, ratio_type, inplace=True)
        return det_objects
    
    def __call__(self, image: khandy.KArray, **kwargs) -> DetObjects:
        det_objects = self.forward(image, **kwargs)
        if self.class_names is not None:
            det_objects.class_names = [self.class_names[ind.item()] for ind in det_objects.classes]
        if self.sort_by is not None:
            det_objects = det_objects.sort(self.sort_by, self.sort_dir, inplace=True)
        return det_objects
    

Index2LabelType = Mapping[Union[int, str], str]


def label_image_by_detector(
    detector: BaseDetector,
    image: np.ndarray,
    index2label: Optional[Index2LabelType] = None,
    **detector_kwargs
) -> khandy.label.DetectIrRecord:
    image_height, image_width = image.shape[:2]
    ir_record = khandy.label.DetectIrRecord('', image_width, image_height)
    det_objects = detector(image, **detector_kwargs)
    for det_object in det_objects:
        if index2label is None:
            label = f'{det_object.class_index}'
        else:
            label = index2label.get(det_object.class_index)
        if label is None:
            continue
        ir_object = khandy.label.DetectIrObject(label, det_object.x_min, det_object.y_min, 
                                                det_object.x_max, det_object.y_max)
        ir_record.objects.append(ir_object)

    return ir_record


def _concatenate_arrays_or_sequences(
    arrays_or_sequences: Union[List[khandy.KArray], List[Sequence]]
) -> Union[khandy.KArray, Sequence]:
    assert len(arrays_or_sequences) > 0
    if khandy.is_list_of(arrays_or_sequences, torch.Tensor):
        return torch.vstack(arrays_or_sequences)
    elif khandy.is_list_of(arrays_or_sequences, np.ndarray):
        return np.vstack(arrays_or_sequences)
    elif khandy.is_list_of(arrays_or_sequences, Sequence):
        first_type = type(arrays_or_sequences[0])
        return first_type(itertools.chain.from_iterable(arrays_or_sequences))
    else:
        raise TypeError('Unsupported type!')
    
    
def concat_det_objects(det_objects_list: List[DetObjects]) -> DetObjects:
    name_to_list = {}
    for det_objects in det_objects_list:
        for name in det_objects.get_fields():
            name_to_list.setdefault(name, []).append(getattr(det_objects, name))
    name_to_sequence = {}
    for name, values in name_to_list.items():
        name_to_sequence[name] = _concatenate_arrays_or_sequences(values)
    return DetObjects(**name_to_sequence)


def detect_in_det_objects(
    detector: BaseDetector, 
    image: np.ndarray, 
    det_objects: DetObjects, 
    min_area: Optional[Union[int, float]] = None,
    **detector_kwargs
) -> DetObjects:
    dst_det_objects_list = []
    for det_object in det_objects:
        if min_area is not None and det_object.area < min_area:
            continue
        
        x_min = round(det_object.x_min)
        y_min = round(det_object.y_min)
        x_max = round(det_object.x_max)
        y_max = round(det_object.y_max)
        cropped = khandy.crop_image(image, x_min, y_min, x_max, y_max)
        objects_in_object = detector(cropped, **detector_kwargs)
        objects_in_object.boxes[:, 0] += det_object.x_min
        objects_in_object.boxes[:, 1] += det_object.y_min
        objects_in_object.boxes[:, 2] += det_object.x_min
        objects_in_object.boxes[:, 3] += det_object.y_min
        dst_det_objects_list.append(objects_in_object)
    return concat_det_objects(dst_det_objects_list)

