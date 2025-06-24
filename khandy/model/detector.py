import copy
import itertools
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, Callable

import numpy as np

import khandy

torch = khandy.import_torch()

__all__ = ['DetObjectItem', 'DetObjectSortDir', 'DetObjectSortBy', 'DetObjects', 
           'BaseDetector', 'convert_det_objects_to_detect_ir_record', 
           'convert_detect_ir_record_to_det_objects',
           'convert_det_objects_to_detect_ir', 'convert_detect_ir_to_det_objects',
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
    _extra_fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in self._extra_fields.items():
            assert name not in self.__annotations__, f'Extra field {name} conflicts with existing field'
            assert len(value) == 1, f'Extra field {name} must have length 1, got {len(value)}'
            
    def __getattr__(self, name: str) -> Any:
        try:
            return self._extra_fields[name]
        except KeyError:
            # NB: use super().__getattribute__ instead of super().__getattr__
            # type object 'object' has no attribute '__getattr__'
            return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self.__annotations__ and name not in ['area']:
            assert len(value) == 1, f'Extra field {name} must have length 1, got {len(value)}'
            self._extra_fields[name] = value
        else:
            super().__setattr__(name, value)
            
    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def to_det_objects(self) -> "DetObjects":
        box = [self.x_min, self.y_min, self.x_max, self.y_max]
        kwargs = {
            'boxes': [box],
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
    
    def __getitem__(self, key: Union[int, slice]) -> Union["DetObjects", DetObjectItem]:
        item = super().__getitem__(key)
        if type(key) == int:
            _extra_fields = {name: getattr(self, name) for name in self._fields 
                            if name not in ['boxes', 'confs', 'classes', 'class_names']}
            return DetObjectItem(
                x_min=item.boxes[0, 0].item(),
                y_min=item.boxes[0, 1].item(),
                x_max=item.boxes[0, 2].item(),
                y_max=item.boxes[0, 3].item(),
                conf=item.confs[0].item(),
                class_index=item.classes[0].item(),
                class_name=item.class_names[0],
                _extra_fields=_extra_fields
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
        interested: Optional[Union[Tuple[int, ...], List[int]]] = None,
        ignored: Optional[Union[Tuple[int, ...], List[int]]] = None,
        inplace: bool = False
    ) -> "DetObjects":
        if interested is not None and ignored is not None:
            raise ValueError("You cannot specify both 'interested' and 'ignored' at the same time.")
        if interested is None and ignored is None:
            return self if inplace else copy.deepcopy(self)
        
        if ignored is not None:
            mask = np.isin(self.classes, ignored, invert=True)
        elif interested is not None:
            mask = np.isin(self.classes, interested)

        keep = np.nonzero(mask)[0]
        return self.filter(keep, inplace)
    
    def filter_by_class_names(
        self, 
        interested: Optional[Union[Tuple[str, ...], List[str]]] = None,
        ignored: Optional[Union[Tuple[str, ...], List[str]]] = None, 
        inplace: bool = False
    ) -> "DetObjects":
        if interested is not None and ignored is not None:
            raise ValueError("You cannot specify both 'interested' and 'ignored' at the same time.")
        if interested is None and ignored is None:
            return self if inplace else copy.deepcopy(self)
        
        if ignored is not None:
            mask = np.isin(self.class_names, ignored, invert=True)
        elif interested is not None:
            mask = np.isin(self.class_names, interested)

        keep = np.nonzero(mask)[0]
        return self.filter(keep, inplace)

    def filter_by_min_area(
        self, 
        min_area: Union[int, float], 
        inplace: bool = False
    ) -> "DetObjects":
        assert isinstance(self.boxes, np.ndarray)
        widths = self.boxes[:, 2] - self.boxes[:, 0]
        heights = self.boxes[:, 3] - self.boxes[:, 1] 
        mask = widths * heights >= min_area
        keep = np.nonzero(mask)[0]
        return self.filter(keep, inplace)

    def filter_by_min_size(
        self, 
        min_width: Optional[Union[int, float]] = None, 
        min_height: Optional[Union[int, float]] = None,
        inplace: bool = False
    ) -> "DetObjects":
        assert isinstance(self.boxes, np.ndarray)
        keep = khandy.filter_small_boxes(self.boxes, min_width, min_height)
        return self.filter(keep, inplace)

    def filter_by_conf(
        self, 
        conf_thresh: Optional[Union[float, List[float], Tuple[float], np.ndarray]], 
        inplace: bool = False
    ) -> "DetObjects":
        assert isinstance(self.confs, np.ndarray)
        if isinstance(conf_thresh, (list, tuple, np.ndarray)):
            conf_thresh = np.array(conf_thresh)
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


def _concatenate_arrays_or_sequences(
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
    
    
def concat_det_objects(det_objects_list: List[DetObjects], only_common_fields: bool = False) -> DetObjects:
    """Concatenates a list of DetObjects into a single DetObjects instance.
    Args:
        det_objects_list (List[DetObjects]): A list of DetObjects instances to concatenate.
        only_common_fields (bool, optional): If True, only the fields common to all DetObjects will be concatenated.
            If False, all DetObjects must have exactly the same fields. Defaults to False.

    Returns:
        DetObjects: A new DetObjects instance with concatenated fields.

    Raises:
        ValueError: If only_common_fields is False and the DetObjects do not have identical fields.
    """
    if len(det_objects_list) == 0:
        common_fields = set()
    else:
        all_fields = [set(det_objects.get_fields()) for det_objects in det_objects_list]
        if only_common_fields:
            common_fields = set.intersection(*all_fields)
        else:
            if not all(fields == all_fields[0] for fields in all_fields):
                raise ValueError("All DetObjects must have the same fields when only_common_fields is False.")
            common_fields = all_fields[0]
    
    name_to_list = {}
    for det_objects in det_objects_list:
        for name in common_fields:
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

