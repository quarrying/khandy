from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Mapping, Optional, Union

import numpy as np

import khandy
torch = khandy.import_torch()

__all__ = ['DetObjectData', 'DetObjectSortDir', 'DetObjectSortBy', 'DetObjects', 
           'BaseDetector', 'Index2LabelType', 'label_image_by_detector']


@dataclass
class DetObjectData:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    conf: float
    class_index: int
    class_name: str
    
    
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

    def __init__(self, boxes: khandy.KArray, confs: Optional[khandy.KArray] = None, 
                 classes: Optional[khandy.KArray] = None, class_names: Optional[List[str]] = None, **kwargs):
        if confs is None:
            if torch is not None and isinstance(boxes, torch.Tensor):
                confs = torch.ones((boxes.shape[0], 1), dtype=torch.float32, device=boxes.device)
            elif isinstance(boxes, np.ndarray):
                confs = np.ones((boxes.shape[0], 1), dtype=np.float32)
            else:
                raise TypeError(f'Unsupported type for confs, got {type(confs)}')
        if confs.ndim == 1:
            confs = confs.reshape((-1, 1))

        if classes is None:
            if torch is not None and isinstance(boxes, torch.Tensor):
                classes = torch.zeros((boxes.shape[0], 1), dtype=torch.int32, device=boxes.device)
            elif isinstance(boxes, np.ndarray):
                classes = np.zeros((boxes.shape[0], 1), dtype=np.int32)
            else:
                raise TypeError(f'Unsupported type for classes, got {type(classes)}')
        if classes.ndim == 1:
            classes = classes.reshape((-1, 1))

        if class_names is None:
            class_names = [f'unnamed_class#{class_ind}' for class_ind in classes.flatten()]

        assert boxes.ndim == confs.ndim == classes.ndim == 2, f'{boxes.ndim} vs {confs.ndim} vs {classes.ndim}'
        assert boxes.shape[1] == 4 and confs.shape[1] == classes.shape[1] == 1
        assert khandy.is_seq_of(class_names, str)
        super().__init__(boxes=boxes, confs=confs, classes=classes, class_names=class_names, **kwargs)

    def __getitem__(self, key: Union[int, slice]) -> Union["DetObjects", DetObjectData]:
        item = super().__getitem__(key)
        if type(key) == int:
            return DetObjectData(
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
        assert isinstance(self.classes, np.ndarray)
        mask = np.zeros((len(self.classes),), dtype=bool)
        for class_ind in interested_class_inds:
            mask = np.logical_or(mask, self.classes[:, 0] == class_ind)
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
        num_classes: int = 1, 
        conf_thresh: float = 0.5, 
        min_width: Optional[Union[float, int]] = None, 
        min_height: Optional[Union[float, int]] = None, 
        min_area: Optional[Union[float, int]] = None, 
        class_names: Optional[Union[List[str], Tuple[str]]] = None, 
        sort_by: Optional[DetObjectSortBy] = None, 
        sort_dir: Optional[DetObjectSortDir] = DetObjectSortDir.DESC
    ):
        self._num_classes = num_classes
        self._conf_thresh = conf_thresh
        self._min_width = min_width
        self._min_height = min_height
        self._min_area = min_area
        self._class_names = class_names
        self._sort_by = sort_by
        self._sort_dir = sort_dir

    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    @property
    def conf_thresh(self) -> Union[float, np.ndarray]:
        return self._conf_thresh
    
    @conf_thresh.setter
    def conf_thresh(self, value: Union[float, List, Tuple, np.ndarray]):
        if isinstance(value, float):
            pass
        elif isinstance(value, (list, tuple)):
            assert khandy.is_seq_of(value, float) and len(value) == self.num_classes
            value = np.array(value)
        elif isinstance(value, np.ndarray):
            assert value.shape == (self.num_classes,) or value.shape == (self.num_classes, 1)
        else:
            raise TypeError(f'unsupported type, got {type(value)}')
        self._conf_thresh = value
    
    @property
    def min_width(self) -> Optional[Union[int, float]]:
        return self._min_width
    
    @min_width.setter
    def min_width(self, value: Optional[Union[int, float]]):
        self._min_width = value
        
    @property
    def min_height(self) -> Optional[Union[int, float]]:
        return self._min_height
    
    @min_height.setter
    def min_height(self, value: Optional[Union[int, float]]):
        self._min_height = value
        
    @property
    def min_area(self) -> Optional[Union[int, float]]:
        return self._min_area
    
    @min_area.setter
    def min_area(self, value: Optional[Union[int, float]]):
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
    
    def __call__(self, image: khandy.KArray, **kwargs) -> DetObjects:
        det_objects = self.forward(image, **kwargs)
        if self.min_width is not None or self.min_height is not None:
            det_objects = det_objects.filter_by_min_size(self.min_width, self.min_height, inplace=True)
        if self.min_area is not None:
            det_objects = det_objects.filter_by_min_area(self.min_area, inplace=True)
        if self.class_names is not None:
            det_objects.class_names = [self.class_names[ind.item()] for ind in det_objects.classes]
        if self.sort_by is not None:
            det_objects = det_objects.sort(self.sort_by, self.sort_dir, inplace=True)
        return det_objects
    

Index2LabelType = Mapping[Union[int, str], str]


def label_image_by_detector(detector: BaseDetector, image: np.ndarray,
                            index2label: Optional[Index2LabelType] = None,
                            **detector_args) -> khandy.label.DetectIrRecord:
    image_height, image_width = image.shape[:2]
    ir_record = khandy.label.DetectIrRecord('', image_width, image_height)
    detect_objs = detector(image, **detector_args)
    for detect_obj in detect_objs:
        if index2label is None:
            label = f'{detect_obj.class_index}'
        else:
            label = index2label.get(detect_obj.class_index)
        if label is None:
            continue
        ir_object = khandy.label.DetectIrObject(label, detect_obj.x_min, detect_obj.y_min, 
                                                detect_obj.x_max, detect_obj.y_max)
        ir_record.objects.append(ir_object)

    return ir_record
