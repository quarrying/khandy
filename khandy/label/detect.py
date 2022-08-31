import copy
import json
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List
import xml.etree.ElementTree as ET

import khandy
import lxml
import lxml.builder
import numpy as np

__all__ = ['PascalVocSource', 'PascalVocSize', 'PascalVocBndbox', 
           'PascalVocObject', 'PascalVocInfo', 'PascalVocHandler',
           'LabelmeShape', 'LabelmeInfo', 'LabelmeHandler',
           'YoloObject', 'YoloInfo', 'YoloHandler', 
           'convert_pascal_voc_to_labelme', 'convert_labelme_to_pascal_voc',
           'convert_labelme_to_yolo', 'convert_yolo_to_labelme',
           'convert_pascal_voc_to_yolo', 'convert_yolo_to_pascal_voc']

@dataclass
class PascalVocSource:
    database: str = ''
    annotation: str = ''
    image: str = ''
    
    
@dataclass
class PascalVocSize:
    height: int
    width: int
    depth: int
    
    
@dataclass
class PascalVocBndbox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    
    
@dataclass
class PascalVocObject:
    name: str
    pose: str = 'Unspecified'
    truncated: int = 0
    difficult: int = 0
    bndbox: Optional[PascalVocBndbox] = None
    
    
@dataclass
class PascalVocInfo:
    folder: str = ''
    filename: str = ''
    path: str = ''
    source: PascalVocSource = PascalVocSource()
    size: Optional[PascalVocSize] = None
    segmented: int = 0
    object: List[PascalVocObject] = field(default_factory=list)
    
    
class PascalVocHandler:
    @staticmethod
    def load(filename) -> PascalVocInfo:
        pascal_voc_info = PascalVocInfo()
        
        xml_tree = ET.parse(filename)
        pascal_voc_info.folder = xml_tree.find('folder').text
        pascal_voc_info.filename = xml_tree.find('filename').text
        pascal_voc_info.path = xml_tree.find('path').text
        pascal_voc_info.segmented = xml_tree.find('segmented').text
        
        source_tag = xml_tree.find('source')
        pascal_voc_info.source = PascalVocSource(
            database=source_tag.find('database').text,
            # annotation=source_tag.find('annotation').text,
            # image=source_tag.find('image').text
        )
        
        size_tag = xml_tree.find('size')
        pascal_voc_info.size = PascalVocSize(
            width=int(size_tag.find('width').text),
            height=int(size_tag.find('height').text),
            depth=int(size_tag.find('depth').text)
        )
        
        object_tags = xml_tree.findall('object')
        for index, obj in enumerate(object_tags):
            bndbox_tag = obj.find('bndbox')
            bndbox = PascalVocBndbox(
                xmin=float(bndbox_tag.find('xmin').text) - 1,
                ymin=float(bndbox_tag.find('ymin').text) - 1,
                xmax=float(bndbox_tag.find('xmax').text) - 1,
                ymax=float(bndbox_tag.find('ymax').text) - 1
            )
            one_object = PascalVocObject(
                name=obj.find('name').text,
                pose=obj.find('pose').text,
                truncated=obj.find('truncated').text,
                difficult=obj.find('difficult').text,
                bndbox=bndbox
            )
            pascal_voc_info.object.append(one_object)
        return pascal_voc_info
        
    @staticmethod
    def save(filename, pascal_voc_info: PascalVocInfo):
        maker = lxml.builder.ElementMaker()
        xml = maker.annotation(
            maker.folder(pascal_voc_info.folder),
            maker.filename(pascal_voc_info.filename),
            maker.path(pascal_voc_info.path),
            maker.source(
                maker.database(pascal_voc_info.source.database),
            ),
            maker.size( 
                maker.width(str(pascal_voc_info.size.width)),
                maker.height(str(pascal_voc_info.size.height)),
                maker.depth(str(pascal_voc_info.size.depth)),
            ),
            maker.segmented(str(pascal_voc_info.segmented)),
        )
        
        for one_object in pascal_voc_info.object:
            object_tag = maker.object(
                maker.name(one_object.name),
                maker.pose(one_object.pose),
                maker.truncated(str(one_object.truncated)),
                maker.difficult(str(one_object.difficult)),
                maker.bndbox(
                    maker.xmin(str(float(one_object.bndbox.xmin))),
                    maker.ymin(str(float(one_object.bndbox.ymin))),
                    maker.xmax(str(float(one_object.bndbox.xmax))),
                    maker.ymax(str(float(one_object.bndbox.ymax))),
                ),
            )
            xml.append(object_tag)
            
        with open(filename, 'wb') as f:
            f.write(lxml.etree.tostring(xml, pretty_print=True, encoding='utf-8'))
        

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8,
                              np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclass
class LabelmeShape:
    label: str
    points: np.ndarray
    shape_type: str
    flags: dict = field(default_factory=dict)
    group_id: Optional[int] = None

    def __post_init__(self):
        self.points = np.asarray(self.points)


@dataclass
class LabelmeInfo:
    version: str = '4.5.6'
    flags: dict = field(default_factory=dict)
    shapes: List[LabelmeShape] = field(default_factory=list)
    imagePath: Optional[str] = None
    imageData: Optional[str] = None
    imageHeight: Optional[int] = None
    imageWidth: Optional[int] = None

    def __post_init__(self):
        for k, shape in enumerate(self.shapes):
            self.shapes[k] = LabelmeShape(**shape)


class LabelmeHandler:
    @staticmethod
    def load(filename) -> LabelmeInfo:
        json_content = khandy.load_json(filename)
        return LabelmeInfo(**json_content)

    @staticmethod
    def save(filename, labelme_info: LabelmeInfo):
        json_content = dataclasses.asdict(labelme_info)
        khandy.save_json(filename, json_content, cls=NumpyEncoder)


@dataclass
class YoloObject:
    label: str
    x_center: float
    y_center: float
    width: float
    height: float
    
    
@dataclass
class YoloInfo:
    image_filename: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    objects: List[YoloObject] = field(default_factory=list)
    
    
class YoloHandler:
    @staticmethod
    def load(filename, **kwargs) -> YoloInfo:
        records = khandy.load_list(filename)

        yolo_info = YoloInfo(
            image_filename=kwargs.get('image_filename'),
            width=kwargs.get('width'),
            height=kwargs.get('height'))
        for record in records:
            record_parts = record.split()
            
            yolo_info.objects.append(YoloObject(
                label=record_parts[0],
                x_center=float(record_parts[1]),
                y_center=float(record_parts[2]),
                width=float(record_parts[3]),
                height=float(record_parts[4]),
            ))
        return yolo_info

    @staticmethod
    def save(filemame, yolo_info: YoloInfo):
        records = []
        for object in yolo_info.objects:
            records.append(f'{object.label} {object.x_center} {object.y_center} {object.width} {object.height}')
        khandy.save_list(filemame, records)

    @staticmethod
    def replace_label(yolo_info: YoloInfo, label_map):
        dst_yolo_info = copy.deepcopy(yolo_info)
        for object in dst_yolo_info.objects:
            object.label= label_map[object.label]
        return dst_yolo_info


def convert_pascal_voc_to_labelme(pascal_voc_info: PascalVocInfo) -> LabelmeInfo:
    labelme_info = LabelmeInfo(
        imagePath=pascal_voc_info.filename,
        imageWidth=pascal_voc_info.size.width,
        imageHeight=pascal_voc_info.size.height
    )
    for object in pascal_voc_info.object:
        labelme_shape = LabelmeShape(
            label=object.name,
            shape_type='rectangle',
            points=[[object.bndbox.xmin, object.bndbox.ymin], 
                    [object.bndbox.xmax, object.bndbox.ymax]]
        )
        labelme_info.shapes.append(labelme_shape)
    return labelme_info


def convert_labelme_to_pascal_voc(labelme_info: LabelmeInfo) -> PascalVocInfo:
    pascal_voc_info = PascalVocInfo(
        filename=labelme_info.imagePath,
        size=PascalVocSize(
            width=labelme_info.imageWidth,
            height=labelme_info.imageHeight,
            depth=3
        )
    )
    for shape in labelme_info.shapes:
        if shape.shape_type != 'rectangle':
            continue
        pascal_voc_object = PascalVocObject(
            name=shape.label,
            bndbox=PascalVocBndbox(
                xmin=shape.points[0][0],
                ymin=shape.points[0][1],
                xmax=shape.points[1][0],
                ymax=shape.points[1][1],
            )
        )
        pascal_voc_info.object.append(pascal_voc_object)
    return pascal_voc_info


def convert_labelme_to_yolo(labelme_info: LabelmeInfo) -> YoloInfo:
    yolo_info = YoloInfo(
        image_filename=labelme_info.imagePath,
        width=labelme_info.imageWidth,
        height=labelme_info.imageHeight
    )
    for shape in labelme_info.shapes:
        if shape.shape_type != 'rectangle':
            continue
        x_center = (shape.points[0][0] + shape.points[1][0]) / (2 * labelme_info.imageWidth)
        y_center = (shape.points[0][1] + shape.points[1][1]) / (2 * labelme_info.imageHeight)
        width = abs(shape.points[0][0] - shape.points[1][0]) / labelme_info.imageWidth
        height = abs(shape.points[0][1] - shape.points[1][1]) / labelme_info.imageHeight
        yolo_object = YoloObject(
            label=shape.label,
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
        )
        yolo_info.objects.append(yolo_object)
    return yolo_info
    
    
def convert_yolo_to_labelme(yolo_info: YoloInfo) -> LabelmeInfo:
    assert (yolo_info.width is not None) and (yolo_info.height is not None)

    labelme_info = LabelmeInfo(
        imagePath=yolo_info.image_filename,
        imageHeight=yolo_info.height,
        imageWidth=yolo_info.width,
    )
    for object in yolo_info.objects:
        x_min = (object.x_center - 0.5 * object.width) * yolo_info.width
        y_min = (object.y_center - 0.5 * object.height) * yolo_info.height
        x_max = (object.x_center + 0.5 * object.width) * yolo_info.width
        y_max = (object.y_center + 0.5 * object.height) * yolo_info.height
        labelme_shape = LabelmeShape(
            label=object.label,
            shape_type='rectangle',
            points=[[x_min, y_min], [x_max, y_max]]
        )
        labelme_info.shapes.append(labelme_shape)
    return labelme_info

    
def convert_pascal_voc_to_yolo(pascal_voc_info: PascalVocInfo) -> YoloInfo:
    yolo_info = YoloInfo(
        image_filename=pascal_voc_info.filename,
        width=pascal_voc_info.size.width,
        height=pascal_voc_info.size.height
    )
    for object in pascal_voc_info.object:
        x_center = (object.bndbox.xmax + object.bndbox.xmin) / (2 * pascal_voc_info.size.width)
        y_center = (object.bndbox.ymax + object.bndbox.ymin) / (2 * pascal_voc_info.size.height)
        width = abs(object.bndbox.xmax - object.bndbox.xmin) / pascal_voc_info.size.width
        height = abs(object.bndbox.ymax - object.bndbox.ymin) / pascal_voc_info.size.height
        yolo_object = YoloObject(
            label=object.name,
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
        )
        yolo_info.objects.append(yolo_object)
    return yolo_info
    
    
def convert_yolo_to_pascal_voc(yolo_info: YoloInfo) -> PascalVocInfo:
    pascal_voc_info = PascalVocInfo(
        filename=yolo_info.image_filename,
        size=PascalVocSize(
            width=yolo_info.width,
            height=yolo_info.height,
            depth=3
        )
    )
    for object in yolo_info.objects:
        x_min = (object.x_center - 0.5 * object.width) * yolo_info.width
        y_min = (object.y_center - 0.5 * object.height) * yolo_info.height
        x_max = (object.x_center + 0.5 * object.width) * yolo_info.width
        y_max = (object.y_center + 0.5 * object.height) * yolo_info.height
        voc_object = PascalVocObject(
            name=object.label,
            bndbox=PascalVocBndbox(xmin=x_min,ymin=y_min,xmax=x_max,ymax=y_max)
        )
        pascal_voc_info.object.append(voc_object)
    return pascal_voc_info
    


