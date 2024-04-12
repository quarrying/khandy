import numbers
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np


def is_boxes(arr, on_axis: int = -1, allow_extra: bool = True) -> bool:
    """Check if the array `arr` satisfies the "boxes" condition on a specific axis.
    
    Args:
        arr (np.ndarray): The input array to be checked.
        on_axis (int, optional): The axis on which to check the boxes condition. Defaults to -1, indicating the last axis.
        allow_extra (bool, optional): Whether to allow dimensions greater than 4 on the specified axis. Defaults to True.
        
    Returns:
        bool: True if the dimension on the specified axis is at least 4 (when allow_extra is True) or
            exactly 4 (when allow_extra is False); False otherwise.
  
    Raises:
        np.AxisError: If the `on_axis` is out of bounds for the array `arr`.
        AssertionError: If `on_axis` is not an integer or `allow_extra` is not a boolean.
    """
    assert isinstance(on_axis, numbers.Integral), "on_axis must be an integer."
    assert isinstance(allow_extra, bool), "allow_extra must be an boolean."

    arr = np.array(arr)
    if not (-arr.ndim <= on_axis < arr.ndim):
        raise np.AxisError(on_axis, arr.ndim)
    if allow_extra:
        return arr.shape[on_axis] >= 4
    else:
        return arr.shape[on_axis] == 4
    
    
class BoxFormat(Enum):
    XYXY = 'xyxy'
    XYWH = 'xywh'
    CXCYWH = 'cxcywh'
    

@dataclass
class _BoxFieldInfo:
    name: str
    slices: slice
    included: List[BoxFormat]
    doc: Optional[str] = None
    

class _BoxFieldDescriptor:
    def __init__(self, field_info):
        self.name = field_info.name
        self.slices = field_info.slices
        self.included = field_info.included
        self.__doc__ = field_info.doc

    def _warn_for_include(self, instance, raise_error=False):
        if instance.format not in self.included:
            box_formats = [str(item) for item in self.included]
            if not raise_error:
                warnings.warn(f'to get {self.name}, box format had better to be in {box_formats}, got {instance.format}')
            else:
                raise ValueError(f'to set {self.name}, box format should to be in {box_formats}, got {instance.format}')
                
    def __get__(self, instance, owner):
        # when the attribute is accessed through the owner, instance is None
        if instance is None:
            return self

        self._warn_for_include(instance)
        if instance.format in self.included:
            return instance.get_fields(self.slices)
            
        if instance.format == BoxFormat.XYXY:
            if self.name == 'widths':
                return instance.x_maxs - instance.x_mins
            elif self.name == 'heights':
                return instance.y_maxs - instance.y_mins
            elif self.name == 'x_centers':
                return (instance.x_maxs + instance.x_mins) * 0.5
            elif self.name == 'y_centers':
                return (instance.y_maxs + instance.y_mins) * 0.5
            elif self.name == 'lengths':
                return instance.maxs - instance.mins
            elif self.name == 'centers':
                return (instance.maxs + instance.mins) * 0.5
        elif instance.format == BoxFormat.XYWH:
            if self.name == 'x_maxs':
                return instance.x_mins + instance.widths
            elif self.name == 'y_maxs':
                return instance.y_mins + instance.heights
            elif self.name == 'x_centers':
                return instance.x_mins + instance.widths * 0.5
            elif self.name == 'y_centers':
                return instance.y_mins + instance.heights * 0.5
            elif self.name == 'maxs':
                return instance.mins + instance.lengths
            elif self.name == 'centers':
                return instance.mins + instance.lengths * 0.5
            elif self.name == 'x_coords':
                return np.concatenate([instance.x_mins, instance.x_maxs], instance.on_axis)
            elif self.name == 'y_coords':
                return np.concatenate([instance.y_mins, instance.y_maxs], instance.on_axis)
        elif instance.format == BoxFormat.CXCYWH:
            if self.name == 'x_mins':
                return instance.x_centers - instance.widths * 0.5
            elif self.name == 'y_mins':
                return instance.y_centers - instance.heights * 0.5
            elif self.name == 'x_maxs':
                return instance.x_centers + instance.widths * 0.5
            elif self.name == 'y_maxs':
                return instance.y_centers + instance.heights * 0.5
            elif self.name == 'mins':
                return instance.centers - instance.lengths * 0.5
            elif self.name == 'maxs':
                return instance.centers + instance.lengths * 0.5
            elif self.name == 'x_coords':
                return np.concatenate([instance.x_mins, instance.x_maxs], instance.on_axis)
            elif self.name == 'y_coords':
                return np.concatenate([instance.y_mins, instance.y_maxs], instance.on_axis)

    def __set__(self, instance, value):
        self._warn_for_include(instance, True) # NB: raise_error is True
        instance.set_fields(self.slices, value)


class Boxes:
    """Boxes is a class used to represent bounding boxes. It contains multiple properties and methods 
    that facilitate the manipulation and processing of bounding box data.
    """
    field_infos = {}
    field_infos['x_mins'] = _BoxFieldInfo('x_mins', slice(0, 1), [BoxFormat.XYXY, BoxFormat.XYWH], 'x coords of top left points')
    field_infos['y_mins'] = _BoxFieldInfo('y_mins', slice(1, 2), [BoxFormat.XYXY, BoxFormat.XYWH], 'y coords of top left points')
    field_infos['x_maxs'] = _BoxFieldInfo('x_maxs', slice(2, 3), [BoxFormat.XYXY], 'x coords of bottom right points')
    field_infos['y_maxs'] = _BoxFieldInfo('y_maxs', slice(3, 4), [BoxFormat.XYXY], 'y coords of bottom right points')
    field_infos['x_centers'] = _BoxFieldInfo('x_centers', slice(0, 1), [BoxFormat.CXCYWH], 'x coords of centers')
    field_infos['y_centers'] = _BoxFieldInfo('y_centers', slice(1, 2), [BoxFormat.CXCYWH], 'y coords of centers')
    field_infos['widths'] = _BoxFieldInfo('widths', slice(2, 3), [BoxFormat.CXCYWH, BoxFormat.XYWH], 'widths of boxes')
    field_infos['heights'] = _BoxFieldInfo('heights', slice(3, 4), [BoxFormat.CXCYWH, BoxFormat.XYWH], 'heights of boxes')
    field_infos['mins'] = _BoxFieldInfo('mins', slice(0, 2), [BoxFormat.XYXY, BoxFormat.XYWH], 'top left points')
    field_infos['maxs'] = _BoxFieldInfo('maxs', slice(2, 4), [BoxFormat.XYXY], 'bottom right points')
    field_infos['centers'] = _BoxFieldInfo('centers', slice(0, 2), [BoxFormat.CXCYWH], 'centers of boxes')
    field_infos['lengths'] = _BoxFieldInfo('lengths', slice(2, 4), [BoxFormat.CXCYWH, BoxFormat.XYWH], 'side lengths of boxes')
    field_infos['x_coords'] = _BoxFieldInfo('x_coords', slice(0, None, 2), [BoxFormat.XYXY], 'x coords of boxes')
    field_infos['y_coords'] = _BoxFieldInfo('y_coords', slice(1, None, 2), [BoxFormat.XYXY], 'y coords of boxes')
    field_infos['boxes'] = _BoxFieldInfo('boxes', slice(0, 4), [BoxFormat.XYXY, BoxFormat.XYWH, BoxFormat.CXCYWH], 'boxes')

    x_mins = x1 =_BoxFieldDescriptor(field_infos['x_mins'])
    y_mins = y1 = _BoxFieldDescriptor(field_infos['y_mins'])
    x_maxs = x2 = _BoxFieldDescriptor(field_infos['x_maxs'])
    y_maxs = y2 = _BoxFieldDescriptor(field_infos['y_maxs'])
    x_centers = cx = _BoxFieldDescriptor(field_infos['x_centers'])
    y_centers = cy = _BoxFieldDescriptor(field_infos['y_centers'])
    widths = w = x_lengths = lx = _BoxFieldDescriptor(field_infos['widths'])
    heights = h = y_lengths = ly = _BoxFieldDescriptor(field_infos['heights'])
    mins = p1 = xy_mins = _BoxFieldDescriptor(field_infos['mins'])
    maxs = p2 = xy_maxs = _BoxFieldDescriptor(field_infos['maxs'])
    centers = c = xy_centers = _BoxFieldDescriptor(field_infos['centers'])
    lengths = l = xy_lengths = _BoxFieldDescriptor(field_infos['lengths'])
    x_coords = x = _BoxFieldDescriptor(field_infos['x_coords'])
    y_coords = y = _BoxFieldDescriptor(field_infos['y_coords'])
    boxes = b = _BoxFieldDescriptor(field_infos['boxes'])

    def __init__(self, boxes, on_axis=-1, format: Union[BoxFormat, str] = BoxFormat.XYXY, 
                 dtype=np.float32, allow_extra=True):
        """Initializes a Boxes object.

        Args:
            boxes: A numpy array that stores box information.
            on_axis: An integer indicating which axis to store the box fields. Default is -1 (the last axis). 
            format: A Union type that can be either a BoxFormat enum or a string representing the format of 
                the boxes. Default value is BoxFormat.XYXY.
            dtype: A numpy dtype that specifies the type of the box fields. Default is np.float32.
            allow_extra (bool, optional): Whether to allow dimensions greater than 4 on the specified axis. Defaults to True.
        """
        data = np.asarray(boxes, dtype=dtype)
        assert is_boxes(data, on_axis, allow_extra)

        self.data = data
        self.on_axis = on_axis
        self.format = BoxFormat(format)
        self.allow_extra = allow_extra

    def __repr__(self):
        format_string = f'{self.__class__.__name__}('
        format_string += f'(numpy.ndarray(shape={self.data.shape}), '
        format_string += f'on_axis={self.on_axis}, '
        format_string += f'format={self.format}, '
        format_string += f'dtype={self.dtype}, '
        format_string += f'allow_extra={self.allow_extra})'
        return format_string
        
    @property
    def count(self):
        """Returns the number of bounding boxes in the data array.
        """
        return self.data.size // self.data.shape[self.on_axis]
        
    @property
    def shape(self):
        """Returns the shape of the data array.
        """
        return self.data.shape
        
    @property
    def dtype(self):
        """Returns the data type of the bounding box coordinates.
        """
        return self.data.dtype
        
    @property
    def areas(self):
        """Calculates and returns the areas of all bounding boxes in the data array.
        """
        return self.widths * self.heights
        
    def copy(self):
        """Returns a copy of the Boxes object with a deep copy of the data array.
        """
        return Boxes(self.data.copy(), on_axis=self.on_axis, format=self.format,
                     dtype=self.dtype, allow_extra=self.allow_extra)
        
    def get_fields(self, indices):
        """Returns specific fields from the bounding boxes based on the provided indices.
        """
        data_slice = [slice(None)] * self.data.ndim
        data_slice[self.on_axis] = indices
        return self.data[tuple(data_slice)] 

    def set_fields(self, indices, value):
        """Set specific fields of the bounding boxes based on the provided indices and value.
        """
        data_slice = [slice(None)] * self.data.ndim
        data_slice[self.on_axis] = indices
        self.data[tuple(data_slice)] = value
        
    def to(self, new_format, copy=True):
        """Converts the bounding box format to the specified new format. 
        Optionally returns a deep copy of the data array if copy is True (default).
        """
        boxes = self.copy() if copy else self
        new_format = BoxFormat(new_format)
        if boxes.format == new_format:
            return boxes

        if (boxes.format, new_format) == (BoxFormat.XYXY, BoxFormat.XYWH):
            boxes.x_maxs -= boxes.x_mins
            boxes.y_maxs -= boxes.y_mins
            boxes.format = BoxFormat.XYWH
        elif (boxes.format, new_format) == (BoxFormat.XYWH, BoxFormat.XYXY):
            boxes.widths += boxes.x_mins
            boxes.heights += boxes.y_mins
            boxes.format = BoxFormat.XYXY
        elif (boxes.format, new_format) == (BoxFormat.XYWH, BoxFormat.CXCYWH):
            boxes.x_mins += boxes.widths * 0.5
            boxes.y_mins += boxes.heights * 0.5
            boxes.format = BoxFormat.CXCYWH
        elif (boxes.format, new_format) == (BoxFormat.CXCYWH, BoxFormat.XYWH):
            boxes.x_centers -= boxes.widths * 0.5
            boxes.y_centers -= boxes.heights * 0.5
            boxes.format = BoxFormat.XYWH
        elif (boxes.format, new_format) == (BoxFormat.XYXY, BoxFormat.CXCYWH):
            boxes.x_maxs -= boxes.x_mins
            boxes.y_maxs -= boxes.y_mins
            boxes.x_mins += boxes.x_maxs * 0.5
            boxes.y_mins += boxes.y_maxs * 0.5
            boxes.format = BoxFormat.CXCYWH
        elif (boxes.format, new_format) == (BoxFormat.CXCYWH, BoxFormat.XYXY):
            boxes.x_centers -= boxes.widths * 0.5
            boxes.y_centers -= boxes.heights * 0.5
            boxes.widths += boxes.x_centers
            boxes.heights += boxes.y_centers
            boxes.format = BoxFormat.XYXY
        return boxes
    
