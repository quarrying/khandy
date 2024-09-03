from abc import ABC, abstractmethod  
from dataclasses import dataclass
from typing import List, Mapping, Optional, Union

import numpy as np

import khandy
torch = khandy.import_torch()

__all__ = ['DetObjectData', 'DetObjects', 'Detector', 'Index2LabelType',
           'label_image_by_detector']


@dataclass
class DetObjectData:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    conf: float
    class_index: int
    class_name: str
    
    
class DetObjects(khandy.EqLenSequences):
    boxes: khandy.KArray
    confs: khandy.KArray
    classes: khandy.KArray
    class_names: List[str]

    def __init__(self, boxes: khandy.KArray, confs: Optional[khandy.KArray] = None, 
                 classes: Optional[khandy.KArray] = None, class_names: Optional[List[str]] = None, **kwargs):
        if confs is None:
            if torch is not None and isinstance(boxes, torch.Tensor):
                confs = torch.ones_like(boxes, dtype=torch.float32)
            elif isinstance(boxes, np.ndarray):
                confs = np.ones_like(boxes, dtype=np.float32)
            else:
                raise TypeError(f'Unsupported type for confs, got {type(confs)}')
        if confs.ndim == 1:
            confs = confs.reshape((-1, 1))

        if classes is None:
            if torch is not None and isinstance(boxes, torch.Tensor):
                classes = torch.zeros_like(boxes, dtype=torch.int32)
            elif isinstance(boxes, np.ndarray):
                classes = np.zeros_like(boxes, dtype=np.int32)
            else:
                raise TypeError(f'Unsupported type for classes, got {type(classes)}')
        if classes.ndim == 1:
            classes = classes.reshape((-1, 1))

        if class_names is None:
            class_names = [f'unnamed_class#{class_ind}' for class_ind in classes]

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
    
    def filter_by_class_index(self, interested_class_inds, inplace=False):
        assert isinstance(self.confs, np.ndarray)
        mask = np.zeros((len(self.classes),), dtype=bool)
        for class_ind in interested_class_inds:
            mask = np.logical_or(mask, self.classes[:, 0] == class_ind)
        return self.filter(mask, inplace)

    def filter_by_min_area(self, min_area, inplace=False):
        assert isinstance(self.confs, np.ndarray)
        widths = self.boxes[:, 2] - self.boxes[:, 0]
        heights = self.boxes[:, 3] - self.boxes[:, 1] 
        mask = widths * heights >= min_area
        keep = np.nonzero(mask)[0]
        return self.filter(keep, inplace)

    def filter_by_min_size(self, min_width, min_height, inplace=False):
        assert isinstance(self.confs, np.ndarray)
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

    def nms(self, iou_thresh, ratio_type='iou', inplace=False):
        assert isinstance(self.confs, np.ndarray)
        keep = khandy.non_max_suppression(self.boxes, self.confs, iou_thresh, self.classes, ratio_type)
        return self.filter(keep, inplace)


class Detector(ABC):
    @abstractmethod
    def __call__(self, image: khandy.KArray, **kwargs) -> DetObjects:
        pass


Index2LabelType = Mapping[Union[int, str], str]


def label_image_by_detector(detector: Detector, image: np.ndarray,
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
