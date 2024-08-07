from abc import ABC, abstractmethod  
from dataclasses import dataclass
from typing import Mapping, Optional, Union

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
    class_index: int
    conf: float
    
    
class DetObjects(khandy.EqLenSequences):
    boxes: khandy.KArray
    confs: khandy.KArray
    classes: khandy.KArray
    
    def __init__(self, boxes: khandy.KArray, confs: khandy.KArray, classes: Optional[khandy.KArray] = None, **kwargs):
        if confs.ndim == 1:
            confs = confs.reshape((-1, 1))
        if classes is None:
            if torch is not None and isinstance(confs, torch.Tensor):
                classes = torch.zeros_like(confs, dtype=torch.int32)
            elif isinstance(confs, np.ndarray):
                classes = np.zeros_like(confs, dtype=np.int32)
            else:
                raise
        if classes.ndim == 1:
            classes = classes.reshape((-1, 1))
        assert boxes.ndim == confs.ndim == classes.ndim == 2, f'{boxes.ndim} vs {confs.ndim} vs {classes.ndim}'
        assert boxes.shape[1] == 4 and confs.shape[1] == classes.shape[1] == 1
        super().__init__(boxes=boxes, confs=confs, classes=classes, **kwargs)

    def __getitem__(self, key: Union[int, slice]) -> Union["DetObjects", DetObjectData]:
        item = super().__getitem__(key)
        if type(key) == int:
            return DetObjectData(
                x_min=item.boxes[0, 0],
                y_min=item.boxes[0, 1],
                x_max=item.boxes[0, 2],
                y_max=item.boxes[0, 3],
                class_index=item.classes[0].item(),
                conf=item.confs[0].item(),
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

    def nms(self, iou_thresh, inplace=False):
        assert isinstance(self.confs, np.ndarray)
        keep = khandy.non_max_suppression(self.boxes, self.confs, iou_thresh, self.classes)
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
