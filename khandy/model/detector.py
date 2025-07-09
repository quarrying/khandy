import copy
import math
import numbers
import itertools
import warnings
import sys
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, List, Literal, Optional, Tuple, Union

if sys.version_info >= (3, 9):
    from collections.abc import Sequence, Mapping, Callable
else:
    from typing import Sequence, Mapping, Callable

import numpy as np

import khandy

torch = khandy.import_torch()

__all__ = ['DetObjectItem', 'DetObjectSortDir', 'DetObjectSortBy', 'DetObjects', 
           'BaseDetector', 'convert_det_objects_to_detect_ir_record', 
           'convert_detect_ir_record_to_det_objects',
           'convert_det_objects_to_detect_ir', 'convert_detect_ir_to_det_objects',
           'concat_det_objects', 'detect_in_det_objects', 'SubsetDetector',
           'get_matches', 'match_det_objects', 'merge_det_objects']


class DetObjectItem:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    conf: float
    class_index: int
    class_name: str

    def __init__(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        conf: float,
        class_index: int,
        class_name: str,
        **kwargs
    ):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.conf = conf
        self.class_index = class_index
        self.class_name = class_name
        self._extra_fields = {}
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._extra_fields[name]
        except KeyError:
            # NB: use super().__getattribute__ instead of super().__getattr__
            # type object 'object' has no attribute '__getattr__'
            return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in (
            "x_min", "y_min", "x_max", "y_max", "conf", 
            "class_index", "class_name", "_extra_fields",
            "x_center", "y_center", "width", "height", "area"
        ):
            if not hasattr(value, '__len__'):
                raise AssertionError(f'Extra field {name} value must have __len__ attribute')
            if len(value) != 1:
                raise AssertionError(f'Extra field {name} value must have length 1, got {len(value)}')
            self._extra_fields[name] = value
        else:
            super().__setattr__(name, value)

    @property
    def x_center(self) -> float:
        return (self.x_min + self.x_max) / 2.0
    
    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2.0
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def to_det_objects(self) -> "DetObjects":
        kwargs = {
            'boxes': [[self.x_min, self.y_min, self.x_max, self.y_max]],
            'confs': [self.conf],
            'classes': [self.class_index],
            'class_names': [self.class_name],
            **self._extra_fields
        }
        return DetObjects(**kwargs)


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
            value = self._set_boxes(value)
        elif name == 'confs':
            value = self._set_confs(value)
        elif name == 'classes':
            value = self._set_classes(value)
        elif name == 'class_names':
            value = self._set_class_names(value)
        super().__setattr__(name, value)

    def _set_boxes(self, boxes: Optional[khandy.KArray] = None) -> khandy.KArray:
        if boxes is None:
            boxes = np.empty((len(self), 4), dtype=np.float32)
        if torch is not None and isinstance(boxes, torch.Tensor):
            pass
        elif isinstance(boxes, np.ndarray):
            pass
        else:
            if len(boxes) == 0:
                boxes = np.empty((0, 4), dtype=np.float32)
            else:
                boxes = np.asarray(boxes, dtype=np.float32)

        assert boxes.ndim == 2, f'boxes ndim is not 2, got {boxes.ndim}'
        assert boxes.shape[1] == 4, f'boxes last axis size is not 4, got {boxes.shape[1]}'
        return boxes

    def _set_confs(self, confs: Optional[khandy.KArray] = None) -> khandy.KArray:
        if isinstance(self.boxes, np.ndarray):
            if confs is None:
                confs = np.ones((len(self),), dtype=np.float32)
            else:
                confs = np.asarray(confs, dtype=np.float32)
        else:
            if confs is None:
                confs = torch.ones((len(self),), dtype=torch.float32, device=self.boxes.device)
            else:
                confs = torch.as_tensor(confs, dtype=torch.float32, device=self.boxes.device)

        if confs.ndim == 2 and confs.shape[-1] == 1:
            confs = confs.squeeze(1)
        assert confs.ndim == 1, f'confs ndim is not 1, got {confs.ndim}'
        return confs

    def _set_classes(self, classes: Optional[khandy.KArray] = None) -> khandy.KArray:
        if isinstance(self.boxes, np.ndarray):
            if classes is None:
                classes = np.zeros((len(self),), dtype=np.int32)
            else:
                classes = np.asarray(classes, dtype=np.int32)
        else:
            if classes is None:
                classes = torch.zeros((len(self),), dtype=torch.int32, device=self.boxes.device)
            else:
                classes = torch.as_tensor(classes, dtype=torch.int32, device=self.boxes.device)

        if classes.ndim == 2 and classes.shape[-1] == 1:
            classes = classes.squeeze(1)
        assert classes.ndim == 1, f'classes ndim is not 1, got {classes.ndim}'
        return classes

    def _set_class_names(self, class_names: Optional[List[str]] = None) -> List[str]:
        if class_names is None:
            class_names = [f'unnamed_class#{class_ind}' for class_ind in self.classes]

        assert khandy.is_seq_of(class_names, str), f'class_names must be list of str'
        return class_names

    def __getitem__(
        self, key: Union[numbers.Integral, slice, List[numbers.Integral], np.ndarray]
    ) -> Union["DetObjects", DetObjectItem]:
        item = super().__getitem__(key)
        if isinstance(key, numbers.Integral):
            _extra_fields = {
                name: getattr(item, name)
                for name in self._fields
                if name not in ["boxes", "confs", "classes", "class_names"]
            }
            return DetObjectItem(
                x_min=item.boxes[0, 0].item(),
                y_min=item.boxes[0, 1].item(),
                x_max=item.boxes[0, 2].item(),
                y_max=item.boxes[0, 3].item(),
                conf=item.confs[0].item(),
                class_index=item.classes[0].item(),
                class_name=item.class_names[0],
                **_extra_fields
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
            mask = np.logical_or(mask, self.classes == class_ind)
        keep = np.nonzero(mask)[0]
        return self.filter(keep, inplace)

    def filter_by_class_indices(
        self,
        interested: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
        ignored: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
        inplace: bool = False
    ) -> "DetObjects":
        if interested is not None and ignored is not None:
            raise ValueError("You cannot specify both 'interested' and 'ignored' at the same time.")
        if interested is None and ignored is None:
            return self if inplace else copy.deepcopy(self)

        if isinstance(interested, int):
            interested = [interested]
        if isinstance(ignored, int):
            ignored = [ignored]

        if ignored is not None:
            mask = np.isin(self.classes, ignored, invert=True)
        elif interested is not None:
            mask = np.isin(self.classes, interested)

        keep = np.nonzero(mask)[0]
        return self.filter(keep, inplace)

    def filter_by_class_names(
        self, 
        interested: Optional[Union[str, Tuple[str, ...], List[str]]] = None,
        ignored: Optional[Union[str, Tuple[str, ...], List[str]]] = None, 
        inplace: bool = False
    ) -> "DetObjects":
        if interested is not None and ignored is not None:
            raise ValueError("You cannot specify both 'interested' and 'ignored' at the same time.")
        if interested is None and ignored is None:
            return self if inplace else copy.deepcopy(self)

        if isinstance(interested, str):
            interested = [interested]
        if isinstance(ignored, str):
            ignored = [ignored]

        if ignored is not None:
            mask = np.isin(self.class_names, ignored, invert=True)
        elif interested is not None:
            mask = np.isin(self.class_names, interested)

        keep = np.nonzero(mask)[0]
        return self.filter(keep, inplace)

    def filter_by_area(
        self, 
        min_area: Union[int, float], 
        inplace: bool = False
    ) -> "DetObjects":
        assert isinstance(self.boxes, np.ndarray)
        keep = khandy.filter_boxes_by_area(self.boxes, min_area)
        return self.filter(keep, inplace)

    def filter_by_min_area(
        self, 
        min_area: Union[int, float], 
        inplace: bool = False
    ) -> "DetObjects":
        warnings.warn(
            "filter_by_min_area is deprecated, use filter_by_area instead.",
            DeprecationWarning
        )
        return self.filter_by_area(min_area, inplace)

    def filter_by_ar(
        self,
        min_ar: Optional[Union[int, float]] = None, 
        max_ar: Optional[Union[int, float]] = None, 
        inplace: bool = False
    )-> "DetObjects":
        assert isinstance(self.boxes, np.ndarray)
        keep = khandy.filter_boxes_by_ar(self.boxes, min_ar, max_ar)
        return self.filter(keep, inplace)

    def filter_by_size(
        self, 
        min_width: Optional[Union[int, float]] = None, 
        min_height: Optional[Union[int, float]] = None,
        inplace: bool = False
    ) -> "DetObjects":
        assert isinstance(self.boxes, np.ndarray)
        keep = khandy.filter_boxes_by_size(self.boxes, min_width, min_height)
        return self.filter(keep, inplace)

    def filter_by_min_size(
        self, 
        min_width: Optional[Union[int, float]] = None, 
        min_height: Optional[Union[int, float]] = None,
        inplace: bool = False
    ) -> "DetObjects":
        warnings.warn(
            "filter_by_min_size is deprecated, use filter_by_size instead.",
            DeprecationWarning
        )
        return self.filter_by_size(min_width, min_height, inplace)

    def filter_by_conf(
        self, 
        conf_thresh: Union[Union[float, List[float], Tuple[float, ...], np.ndarray]], 
        inplace: bool = False
    ) -> "DetObjects":
        assert isinstance(self.confs, np.ndarray)
        if isinstance(conf_thresh, (list, tuple, np.ndarray)):
            conf_thresh = np.asarray(conf_thresh)
            mask = self.confs > conf_thresh[self.classes]
        else:
            mask = self.confs > conf_thresh
        keep = np.nonzero(mask)[0]
        return self.filter(keep, inplace)

    def filter_by_func(
        self, 
        func: Callable[[DetObjectItem], bool], 
        inplace: bool = False
    ) -> "DetObjects":
        assert isinstance(self.boxes, np.ndarray)
        mask = np.array([func(det_object) for det_object in self], dtype=bool)
        keep = np.nonzero(mask)[0]
        return self.filter(keep, inplace)

    def nms(
        self,
        thresh: Union[Union[float, List[float], Tuple[float, ...], np.ndarray]],
        ratio_type: Union[Literal['iou', 'iom'], Sequence[Literal['iou', 'iom']]] = "iou",
        class_agnostic: bool = False,
        inplace: bool = False,
    ) -> "DetObjects":
        assert isinstance(self.confs, np.ndarray)
        if class_agnostic:
            keep = khandy.non_max_suppression(
                self.boxes, self.confs, thresh, None, ratio_type
            )
        else:
            keep = khandy.non_max_suppression(
                self.boxes, self.confs, thresh, self.classes, ratio_type
            )
        return self.filter(keep, inplace)

    def sort(
        self,
        sort_by: DetObjectSortBy,
        direction: DetObjectSortDir = DetObjectSortDir.DESC,
    ) -> "DetObjects":
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
        conf_thresh: Optional[Union[float, List[float], Tuple[float, ...], np.ndarray]] = None,
        min_width: Optional[Union[int, float]] = None,
        min_height: Optional[Union[int, float]] = None,
        min_area: Optional[Union[int, float]] = None,
        class_names: Optional[Union[List[str], Tuple[str, ...]]] = None,
        sort_by: Optional[DetObjectSortBy] = None,
        sort_dir: Optional[DetObjectSortDir] = DetObjectSortDir.DESC,
        **kwargs
    ):
        # TODO: min_width, min_height and min_area can also support sequence types
        # which means that they can be filtered by class.
        if num_classes is not None:
            assert isinstance(num_classes, int) and num_classes > 0, f'num_classes must be a positive integer, got {num_classes}'
        self._num_classes = num_classes

        self.conf_thresh = conf_thresh
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area
        self.class_names = class_names
        self.sort_by = sort_by
        self.sort_dir = sort_dir
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def num_classes(self) -> Optional[int]:
        return self._num_classes
    
    @property
    def conf_thresh(self) -> Optional[Union[float, np.ndarray]]:
        return self._conf_thresh
    
    @conf_thresh.setter
    def conf_thresh(self, value: Optional[Union[float, List[float], Tuple[float, ...], np.ndarray]]):
        if value is None or isinstance(value, float):
            pass
        elif isinstance(value, (list, tuple)):
            assert self.num_classes is not None, 'num_classes must be set before setting conf_thresh'
            assert khandy.is_seq_of(value, float), f'conf_thresh must be a list or tuple of floats, got {type(value)}'
            assert len(value) == self.num_classes, f'conf_thresh must have length {self.num_classes}, got {len(value)}'
            value = np.array(value)
        elif isinstance(value, np.ndarray):
            assert self.num_classes is not None, 'num_classes must be set before setting conf_thresh'
            assert value.dtype in [np.float32, np.float64], f'conf_thresh must be a numpy array of floats, got {value.dtype}'
            assert value.shape == (self.num_classes,) or value.shape == (self.num_classes, 1),\
                f'conf_thresh shape must be ({self.num_classes},) or ({self.num_classes}, 1), got {value.shape}'
            value = value.flatten()
        else:
            raise TypeError(f'Unsupported type for conf_thresh, got {type(value)}')
        
        if value is not None:
            assert np.min(value) >= 0, f'conf_thresh must be >= 0, got {np.min(value)}'
            assert np.max(value) <= 1, f'conf_thresh must be <= 1, got {np.max(value)}'
            
        self._conf_thresh = value
    
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
    def class_names(self, value: Optional[Union[List[str], Tuple[str, ...]]]):
        if value is None:
            pass
        elif isinstance(value, (list, tuple)):
            assert self.num_classes is not None, 'num_classes must be set before setting class_names'
            assert khandy.is_seq_of(value, str), f'class_names must be a list or tuple of strings, got {type(value)}'
            assert len(value) == self.num_classes, f'class_names must have length {self.num_classes}, got {len(value)}'
            value = list(value)
        else:
            raise TypeError(f'Unsupported type for class_names, got {type(value)}')
        self._class_names = value
        
    @property
    def sort_by(self) -> Optional[DetObjectSortBy]:
        return self._sort_by
    
    @sort_by.setter
    def sort_by(self, value: Optional[DetObjectSortBy]):
        assert value is None or isinstance(value, DetObjectSortBy), f'Unsupported type for sort_by, got {type(value)}'
        self._sort_by = value
        
    @property
    def sort_dir(self) -> DetObjectSortDir:
        return self._sort_dir
    
    @sort_dir.setter
    def sort_dir(self, value: Optional[DetObjectSortDir]):
        assert value is None or isinstance(value, DetObjectSortDir), f'Unsupported type for sort_dir, got {type(value)}'
        self._sort_dir = value or DetObjectSortDir.DESC

    @abstractmethod
    def forward(self, image: khandy.KArray, **kwargs) -> DetObjects:
        pass
    
    def filter_by_conf(self, det_objects: DetObjects, inplace: bool = False) -> DetObjects:
        if self.conf_thresh is not None:
            return det_objects.filter_by_conf(self.conf_thresh, inplace=inplace)
        return det_objects
    
    def filter_by_size(self, det_objects: DetObjects, inplace: bool = False) -> DetObjects:
        if self.min_width is not None or self.min_height is not None:
            return det_objects.filter_by_size(self.min_width, self.min_height, inplace=inplace)
        return det_objects
    
    def filter_by_area(self, det_objects: DetObjects, inplace: bool = False) -> DetObjects:
        if self.min_area is not None:
            return det_objects.filter_by_area(self.min_area, inplace=inplace)
        return det_objects
    
    def __call__(self, image: khandy.KArray, **kwargs) -> DetObjects:
        det_objects = self.forward(image, **kwargs)
        if self.num_classes is not None and len(det_objects) > 0:
            max_class_index = np.max(det_objects.classes)
            assert max_class_index < self.num_classes, \
                f'max of det_objects.classes must be < self.num_classes ({self.num_classes}), got {max_class_index}'
        if self.class_names is not None:
            det_objects.class_names = [self.class_names[ind.item()] for ind in det_objects.classes]
        if self.sort_by is not None:
            det_objects = det_objects.sort(self.sort_by, self.sort_dir)
        return det_objects


def convert_det_objects_to_detect_ir_record(
    det_objects: DetObjects, 
    image_width: int, 
    image_height: int,
    filename: str = ''
) -> khandy.label.DetectIrRecord:
    warnings.warn(
        "convert_det_objects_to_detect_ir_record is deprecated, use convert_det_objects_to_detect_ir instead.",
        DeprecationWarning
    )
    ir_record = khandy.label.DetectIrRecord(
        filename=filename, 
        width=image_width, 
        height=image_height
    )
    for det_object in det_objects:
        ir_object = khandy.label.DetectIrObject(
            label=det_object.class_name, 
            x_min=det_object.x_min, 
            y_min=det_object.y_min, 
            x_max=det_object.x_max, 
            y_max=det_object.y_max)
        ir_record.objects.append(ir_object)
    return ir_record


def convert_detect_ir_record_to_det_objects(
    ir_record: khandy.label.DetectIrRecord,
    label2index: Optional[Mapping[str, int]] = None
) -> DetObjects:
    warnings.warn(
        "convert_detect_ir_record_to_det_objects is deprecated, use convert_detect_ir_to_det_objects instead.",
        DeprecationWarning
    )
    class_names, boxes = [], []
    for ir_object in ir_record.objects:
        class_names.append(ir_object.label)
        boxes.append([ir_object.x_min, ir_object.y_min, 
                      ir_object.x_max, ir_object.y_max])
    if label2index is not None:
        class_indices = [label2index[name] for name in class_names]
    else:
        class_indices = [-1 for _ in class_names]
    det_objects = DetObjects(
        boxes=boxes, 
        class_names=class_names,
        class_indices=class_indices
    )
    return det_objects


def convert_det_objects_to_detect_ir(
    det_objects: DetObjects, 
) -> List[khandy.label.DetectIrObject]:
    ir_objects = []
    for det_object in det_objects:
        ir_object = khandy.label.DetectIrObject(
            label=det_object.class_name, 
            x_min=det_object.x_min, 
            y_min=det_object.y_min, 
            x_max=det_object.x_max, 
            y_max=det_object.y_max)
        ir_objects.append(ir_object)
    return ir_objects


def convert_detect_ir_to_det_objects(
    detect_ir: Union[khandy.label.DetectIrRecord, 
                     khandy.label.DetectIrObject, 
                     List[khandy.label.DetectIrObject]],
    label2index: Optional[Mapping[str, int]] = None
) -> DetObjects:
    if isinstance(detect_ir, khandy.label.DetectIrRecord):
        ir_objects = detect_ir.objects
    elif isinstance(detect_ir, khandy.label.DetectIrObject):
        ir_objects = [detect_ir]
    else:
        ir_objects = detect_ir
        
    class_names, boxes = [], []
    for ir_object in ir_objects:
        class_names.append(ir_object.label)
        boxes.append([ir_object.x_min, ir_object.y_min, 
                      ir_object.x_max, ir_object.y_max])
    if label2index is not None:
        class_indices = [label2index[name] for name in class_names]
    else:
        class_indices = [-1 for _ in class_names]
    det_objects = DetObjects(
        boxes=boxes, 
        class_names=class_names,
        class_indices=class_indices
    )
    return det_objects


def _concat_arrays_or_sequences(
    arrays_or_sequences: Union[List[khandy.KArray], List[Sequence]]
) -> Union[khandy.KArray, Sequence]:
    assert len(arrays_or_sequences) > 0
    if khandy.is_torch_available() and khandy.is_list_of(arrays_or_sequences, torch.Tensor):
        return torch.cat(arrays_or_sequences, dim=0)
    elif khandy.is_list_of(arrays_or_sequences, np.ndarray):
        return np.concatenate(arrays_or_sequences, axis=0)
    elif khandy.is_list_of(arrays_or_sequences, Sequence):
        first_type = type(arrays_or_sequences[0])
        return first_type(itertools.chain.from_iterable(arrays_or_sequences))
    else:
        raise TypeError('Unsupported type!')


def concat_det_objects(
    det_objects_list: List[Union[DetObjects, DetObjectItem]],
    only_common_fields: bool = False,
) -> DetObjects:
    """Concatenates a list of DetObjects into a single DetObjects instance.
    Args:
        det_objects_list (List[Union[DetObjects, DetObjectItem]]): A list of DetObjects or DetObjectItem instances to concatenate.
        only_common_fields (bool, optional): If True, only the fields common to all DetObjects will be concatenated.
            If False, all DetObjects must have exactly the same fields. Defaults to False.

    Returns:
        DetObjects: A new DetObjects instance with concatenated fields.

    Raises:
        ValueError: If only_common_fields is False and the DetObjects do not have identical fields.
    """
    inner_det_objects_list = []
    for det_objects in det_objects_list:
        if isinstance(det_objects, DetObjects):
            inner_det_objects_list.append(det_objects)
        elif isinstance(det_objects, DetObjectItem):
            inner_det_objects_list.append(det_objects.to_det_objects())
        else:
            raise TypeError(f'Unsupported type, got {type(det_objects)}')

    if len(inner_det_objects_list) == 0:
        common_fields = set()
    else:
        all_fields = [set(det_objects.get_fields()) for det_objects in inner_det_objects_list]
        if only_common_fields:
            common_fields = set.intersection(*all_fields)
        else:
            if not all(fields == all_fields[0] for fields in all_fields):
                raise ValueError("All DetObjects must have the same fields when only_common_fields is False.")
            common_fields = all_fields[0]

    name_to_list = {}
    for det_objects in inner_det_objects_list:
        for name in common_fields:
            name_to_list.setdefault(name, []).append(getattr(det_objects, name))
    name_to_sequence = {}
    for name, values in name_to_list.items():
        name_to_sequence[name] = _concat_arrays_or_sequences(values)
    return DetObjects(**name_to_sequence)


def detect_in_det_objects(
    detector: BaseDetector, 
    image: np.ndarray, 
    det_objects: DetObjects, 
    **detector_kwargs
) -> DetObjects:
    dst_det_objects_list = []
    for det_object in det_objects:
        x_min = math.floor(det_object.x_min)
        y_min = math.floor(det_object.y_min)
        x_max = math.ceil(det_object.x_max)
        y_max = math.ceil(det_object.y_max)
        if x_min >= x_max or y_min >= y_max:
            warnings.warn(
                f"Skipping detection in object {det_object.class_name} with invalid coordinates: "
                f"({x_min}, {y_min}, {x_max}, {y_max})",
                UserWarning
            )
            continue
        cropped = khandy.crop_image(image, x_min, y_min, x_max, y_max)
        objects_in_object = detector(cropped, **detector_kwargs)
        objects_in_object.boxes[:, 0] += x_min
        objects_in_object.boxes[:, 1] += y_min
        objects_in_object.boxes[:, 2] += x_min
        objects_in_object.boxes[:, 3] += y_min
        dst_det_objects_list.append(objects_in_object)
    return concat_det_objects(dst_det_objects_list)


class SubsetDetector(BaseDetector):
    def __init__(
        self, 
        detector: BaseDetector, 
        interested_class_names: List[str]
    ) -> None:
        assert isinstance(detector, BaseDetector)
        assert detector.class_names is not None
        assert len(interested_class_names) != 0
        assert len(set(interested_class_names)) == len(interested_class_names)
        assert set(interested_class_names).issubset(set(detector.class_names))
        self.detector = detector
        self.interested_class_names = interested_class_names
        
        if detector.conf_thresh is None or isinstance(detector.conf_thresh, float):
            conf_thresh = detector.conf_thresh
        else:
            conf_thresh = [detector.conf_thresh[detector.class_names.index(name)] 
                           for name in interested_class_names]
        super().__init__(
            num_classes=len(self.interested_class_names),
            class_names=interested_class_names,
            conf_thresh=conf_thresh,
            min_width=detector.min_width,
            min_height=detector.min_height,
            min_area=detector.min_area,
            sort_by=detector.sort_by,
            sort_dir=detector.sort_dir
        )
        
    def forward(self, image: khandy.KArray, **kwargs) -> DetObjects:
        det_objects = self.detector(image, **kwargs)
        det_objects.filter_by_class_names(self.interested_class_names, inplace=True)
        det_objects.classes = [self.interested_class_names.index(name) for name in det_objects.class_names]
        return det_objects

    def __call__(self, image: khandy.KArray, **kwargs) -> DetObjects:
        return self.forward(image, **kwargs)


def get_matches(
    overlaps: np.ndarray, 
    thresh: float = 0.5, 
    match_type: Literal['1vn', 'nv1', '1v1'] = '1v1'
) -> np.ndarray:
    """Find matching pairs of boxes based on overlap ratios.

    Args:
        overlaps (np.ndarray): A 2D array of shape `(num_boxes1, num_boxes2)` containing 
            the overlap ratios between two sets of boxes.
        thresh (float, optional): The overlap threshold for considering a pair of boxes 
            as a match. Defaults to 0.5.
        match_type (Literal['1vn', 'nv1', '1v1'], optional): The matching strategy to use:
            - '1vn': One box from the first set can match multiple boxes from the second set.
            - 'nv1': Multiple boxes from the first set can match one box from the second set.
            - '1v1': One-to-one matching between boxes from the two sets.
            Defaults to '1v1'.

    Returns:
        np.ndarray: A 2D array of shape `(num_matches, 3)` where each row contains 
            the index of the box from the first set, the index of the box from the second set, 
            and the overlap ratio between them.
    """
    if match_type.lower() not in ['1vn', 'nv1', '1v1']:
        raise ValueError(f'Invalid match_type: {match_type}')

    num_boxes1, num_boxes2 = overlaps.shape
    if num_boxes1 >= 1 and num_boxes2 >= 1:
        indices = np.nonzero(overlaps >= thresh)
        matches = np.concatenate((np.stack(indices, 1), overlaps[indices][:, None]), 1)
        if match_type.lower() in ['1vn', '1v1']:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        if match_type.lower() in ['nv1', '1v1']:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    else:
        matches = np.zeros((0, 3))
    return matches


def _adjust_boxes(boxes, classes, max_coordinate=None):
    """Adjust the boxes by adding an offset based on their class indices.

    This method is used to perform boxes matching independently per class.
    An offset is added to each box such that boxes from different classes do not overlap.

    Args:
        boxes (np.ndarray): A 2D array of shape `(num_boxes, 4)` containing the bounding boxes.
        classes (np.ndarray): A 1D array of shape `(num_boxes,)` containing the class indices of the boxes.
        max_coordinate (float, optional): The maximum coordinate value among all boxes. 
            If None, it will be calculated from the boxes. Defaults to None.

    Returns:
        np.ndarray: A 2D array of shape `(num_boxes, 4)` containing the adjusted boxes.
    """
    classes = np.asarray(classes)
    if max_coordinate is None:
        max_coordinate = np.max(boxes)
    offsets = classes * (max_coordinate + 1)
    if offsets.ndim == 1:
        offsets = offsets[:, None]
    # cannot use inplace add
    boxes = boxes + offsets
    return boxes


def match_det_objects(
    det_objects1: DetObjects,
    det_objects2: DetObjects,
    thresh: float = 0.5, 
    ratio_type: Literal['iou', 'iom'] = 'iou',
    match_type: Literal['1vn', 'nv1', '1v1'] = '1v1',
    return_missed_inds: bool = False,
    class_agnostic: bool = False
)-> Union[List[Tuple[int, int]], Tuple[List[Tuple[int, int]], List[int], List[int]]]:
    """Match detection objects from two sets based on their overlap ratios.

    Args:
        det_objects1 (DetObjects): The first set of detection objects.
        det_objects2 (DetObjects): The second set of detection objects.
        thresh (float, optional): The overlap threshold for considering a pair of objects as a match.
            Defaults to 0.5.
        ratio_type (Literal['iou', 'iom'], optional): The type of overlap ratio to use, either 'iou' 
            (Intersection over Union) or 'iom' (Intersection over Minimum). Defaults to 'iou'.
        match_type (Literal['1vn', 'nv1', '1v1'], optional): The matching strategy to use:
            - '1vn': One object from the first set can match multiple objects from the second set.
            - 'nv1': Multiple objects from the first set can match one object from the second set.
            - '1v1': One-to-one matching between objects from the two sets.
            Defaults to '1v1'.
        return_missed_inds (bool, optional): If True, return the indices of missed objects in both sets. 
            Defaults to False.
        class_agnostic (bool, optional): If True, perform class-agnostic matching. Defaults to False.

    Returns:
        Union[List[Tuple[int, int]], Tuple[List[Tuple[int, int]], List[int], List[int]]]:
            If return_missed_inds is False, return a list of tuples containing the indices of matched objects.
            If return_missed_inds is True, return a tuple containing the list of matched indices,
            the list of missed indices in det_objects1, and the list of missed indices in det_objects2.
    """
    num_objects1, num_objects2 = len(det_objects1), len(det_objects2)
    if num_objects1 == 0 or num_objects2 == 0:
        if not return_missed_inds:
            return []
        else:
            missed_obj1_inds = list(set(range(num_objects1)))
            missed_obj2_inds = list(set(range(num_objects2)))
            return [], missed_obj1_inds, missed_obj2_inds

    if not class_agnostic:
        max_coordinate = np.max([np.max(det_objects1.boxes), np.max(det_objects2.boxes)])
        boxes1 = _adjust_boxes(det_objects1.boxes, det_objects1.classes, max_coordinate=max_coordinate)
        boxes2 = _adjust_boxes(det_objects2.boxes, det_objects2.classes, max_coordinate=max_coordinate)
    else:
        boxes1, boxes2 = det_objects1.boxes, det_objects2.boxes

    overlaps = khandy.pairwise_overlap_ratio(boxes1, boxes2, ratio_type)
    matches = get_matches(overlaps, thresh, match_type)
    matched_ind_pairs = [(int(obj1_ind), int(obj2_ind)) for obj1_ind, obj2_ind, _ in matches]

    if not return_missed_inds:
        return matched_ind_pairs
    else:
        matched_obj1_inds = [obj1_ind for obj1_ind, obj2_ind in matched_ind_pairs]
        matched_obj2_inds = [obj2_ind for obj1_ind, obj2_ind in matched_ind_pairs]
        missed_obj1_inds = list(set(range(num_objects1)) - set(matched_obj1_inds))
        missed_obj2_inds = list(set(range(num_objects2)) - set(matched_obj2_inds))
        return matched_ind_pairs, missed_obj1_inds, missed_obj2_inds


def merge_det_objects(
    det_objects1: DetObjects,
    det_objects2: DetObjects,
    thresh: float = 0.5,
    ratio_type: Literal['iou', 'iom'] = 'iou',
    match_type: Literal['1vn', 'nv1', '1v1'] = '1v1',
    merge_type: Literal['object1', 'object2', 'max_conf', 'max_area'] = 'max_conf',
    class_agnostic: bool = False
) -> DetObjects:
    """
    Merge two sets of detection objects based on their overlap ratios.

    Args:
        det_objects1 (DetObjects): The first set of detection objects.
        det_objects2 (DetObjects): The second set of detection objects.
        thresh (float, optional): The overlap threshold for considering a pair of objects as a match. 
            Defaults to 0.5.
        ratio_type (Literal['iou', 'iom'], optional): The type of overlap ratio to use, either 'iou' 
            (Intersection over Union) or 'iom' (Intersection over Minimum). Defaults to 'iou'.
        match_type (Literal['1vn', 'nv1', '1v1'], optional): The matching strategy to use:
            - '1vn': One object from the first set can match multiple objects from the second set.
            - 'nv1': Multiple objects from the first set can match one object from the second set.
            - '1v1': One-to-one matching between objects from the two sets.
            Defaults to '1v1'.
        merge_type (Literal['object1', 'object2', 'max_conf', 'max_area'], optional): The merging strategy to use:
            - 'object1': Keep objects from the first set and unmatched objects from the second set.
            - 'object2': Keep objects from the second set and unmatched objects from the first set.
            - 'max_conf': Keep the object with the maximum confidence score from each matched pair.
            - 'max_area': Keep the object with the maximum area from each matched pair.
            Defaults to 'max_conf'.
        class_agnostic (bool, optional): If True, perform class-agnostic matching. Defaults to False.

    Returns:
        DetObjects: A new DetObjects instance containing the merged detection objects.
    """
    if len(det_objects1) == 0 or len(det_objects2) == 0:
        return concat_det_objects([det_objects1, det_objects2], only_common_fields=True)
    
    matched_ind_pairs, missed_obj1_inds, missed_obj2_inds = match_det_objects(
        det_objects1, det_objects2, thresh, ratio_type, match_type, 
        return_missed_inds=True, class_agnostic=class_agnostic)

    if merge_type.lower() in ['object1', 'box1', 'obj1']:
        det_objects_list = [det_objects1, det_objects2[missed_obj2_inds]]
    elif merge_type.lower() in ['object2', 'box2', 'obj2']:
        det_objects_list = [det_objects1[missed_obj1_inds], det_objects2]
    elif merge_type.lower() == 'max_conf':
        det_objects_list = [det_objects1[missed_obj1_inds], det_objects2[missed_obj2_inds]]
        for box1_ind, box2_ind in matched_ind_pairs:
            if det_objects2[box2_ind].conf >= det_objects1[box1_ind].conf:
                det_objects_list.append(det_objects2[box2_ind])
            else:
                det_objects_list.append(det_objects1[box1_ind])
    elif merge_type.lower() == 'max_area':
        det_objects_list = [det_objects1[missed_obj1_inds], det_objects2[missed_obj2_inds]]
        for box1_ind, box2_ind in matched_ind_pairs:
            if det_objects2[box2_ind].area >= det_objects1[box1_ind].area:
                det_objects_list.append(det_objects2[box2_ind])
            else:
                det_objects_list.append(det_objects1[box1_ind])
    else:
        raise ValueError(f'Unsupported merge_type, got {merge_type}')

    return concat_det_objects(det_objects_list, only_common_fields=True)
