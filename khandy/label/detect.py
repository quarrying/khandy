import os
import copy
import json
import dataclasses
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Optional, List
import xml.etree.ElementTree as ET

import khandy
import lxml
import lxml.builder
import numpy as np


__all__ = ['DetectIrObject', 'DetectIrRecord', 'load_detect', 
           'save_detect', 'convert_detect', 'replace_detect_label',
           'load_coco_class_names']


@dataclass
class DetectIrObject:
    """Intermediate Representation Format of Object
    """
    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    
@dataclass
class DetectIrRecord:
    """Intermediate Representation Format of Record
    """
    filename: str
    width: int
    height: int
    objects: List[DetectIrObject] = field(default_factory=list)
    
    
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
class PascalVocRecord:
    folder: str = ''
    filename: str = ''
    path: str = ''
    source: PascalVocSource = PascalVocSource()
    size: Optional[PascalVocSize] = None
    segmented: int = 0
    objects: List[PascalVocObject] = field(default_factory=list)
    
    
class PascalVocHandler:
    @staticmethod
    def load(filename, **kwargs) -> PascalVocRecord:
        pascal_voc_record = PascalVocRecord()
        
        xml_tree = ET.parse(filename)
        pascal_voc_record.folder = xml_tree.find('folder').text
        pascal_voc_record.filename = xml_tree.find('filename').text
        pascal_voc_record.path = xml_tree.find('path').text
        pascal_voc_record.segmented = xml_tree.find('segmented').text
        
        source_tag = xml_tree.find('source')
        pascal_voc_record.source = PascalVocSource(
            database=source_tag.find('database').text,
            # annotation=source_tag.find('annotation').text,
            # image=source_tag.find('image').text
        )
        
        size_tag = xml_tree.find('size')
        pascal_voc_record.size = PascalVocSize(
            width=int(size_tag.find('width').text),
            height=int(size_tag.find('height').text),
            depth=int(size_tag.find('depth').text)
        )
        
        object_tags = xml_tree.findall('object')
        for index, object_tag in enumerate(object_tags):
            bndbox_tag = object_tag.find('bndbox')
            bndbox = PascalVocBndbox(
                xmin=float(bndbox_tag.find('xmin').text) - 1,
                ymin=float(bndbox_tag.find('ymin').text) - 1,
                xmax=float(bndbox_tag.find('xmax').text) - 1,
                ymax=float(bndbox_tag.find('ymax').text) - 1
            )
            pascal_voc_object = PascalVocObject(
                name=object_tag.find('name').text,
                pose=object_tag.find('pose').text,
                truncated=object_tag.find('truncated').text,
                difficult=object_tag.find('difficult').text,
                bndbox=bndbox
            )
            pascal_voc_record.objects.append(pascal_voc_object)
        return pascal_voc_record
        
    @staticmethod
    def save(filename, pascal_voc_record: PascalVocRecord):
        maker = lxml.builder.ElementMaker()
        xml = maker.annotation(
            maker.folder(pascal_voc_record.folder),
            maker.filename(pascal_voc_record.filename),
            maker.path(pascal_voc_record.path),
            maker.source(
                maker.database(pascal_voc_record.source.database),
            ),
            maker.size( 
                maker.width(str(pascal_voc_record.size.width)),
                maker.height(str(pascal_voc_record.size.height)),
                maker.depth(str(pascal_voc_record.size.depth)),
            ),
            maker.segmented(str(pascal_voc_record.segmented)),
        )
        
        for pascal_voc_object in pascal_voc_record.objects:
            object_tag = maker.object(
                maker.name(pascal_voc_object.name),
                maker.pose(pascal_voc_object.pose),
                maker.truncated(str(pascal_voc_object.truncated)),
                maker.difficult(str(pascal_voc_object.difficult)),
                maker.bndbox(
                    maker.xmin(str(float(pascal_voc_object.bndbox.xmin))),
                    maker.ymin(str(float(pascal_voc_object.bndbox.ymin))),
                    maker.xmax(str(float(pascal_voc_object.bndbox.xmax))),
                    maker.ymax(str(float(pascal_voc_object.bndbox.ymax))),
                ),
            )
            xml.append(object_tag)
            
        if not filename.endswith('.xml'):
            filename = filename + '.xml'
        with open(filename, 'wb') as f:
            f.write(lxml.etree.tostring(xml, pretty_print=True, encoding='utf-8'))
            
    @staticmethod
    def to_ir(pascal_voc_record: PascalVocRecord) -> DetectIrRecord:
        ir_record = DetectIrRecord(
            filename=pascal_voc_record.filename,
            width=pascal_voc_record.size.width,
            height=pascal_voc_record.size.height
        )
        for pascal_voc_object in pascal_voc_record.objects:
            ir_object = DetectIrObject(
                label=pascal_voc_object.name,
                x_min=pascal_voc_object.bndbox.xmin,
                y_min=pascal_voc_object.bndbox.ymin,
                x_max=pascal_voc_object.bndbox.xmax,
                y_max=pascal_voc_object.bndbox.ymax
            )
            ir_record.objects.append(ir_object)
        return ir_record
        
    @staticmethod
    def from_ir(ir_record: DetectIrRecord) -> PascalVocRecord:
        pascal_voc_record = PascalVocRecord(
            filename=ir_record.filename,
            size=PascalVocSize(
                width=ir_record.width,
                height=ir_record.height,
                depth=3
            )
        )
        for ir_object in ir_record.objects:
            pascal_voc_object = PascalVocObject(
                name=ir_object.label,
                bndbox=PascalVocBndbox(
                    xmin=ir_object.x_min,
                    ymin=ir_object.y_min,
                    xmax=ir_object.x_max,
                    ymax=ir_object.y_max,
                )
            )
            pascal_voc_record.objects.append(pascal_voc_object)
        return pascal_voc_record
        
        
class _NumpyEncoder(json.JSONEncoder):
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
class LabelmeRecord:
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
    def load(filename, **kwargs) -> LabelmeRecord:
        json_content = khandy.load_json(filename)
        return LabelmeRecord(**json_content)

    @staticmethod
    def save(filename, labelme_record: LabelmeRecord):
        json_content = dataclasses.asdict(labelme_record)
        khandy.save_json(filename, json_content, cls=_NumpyEncoder)

    @staticmethod
    def to_ir(labelme_record: LabelmeRecord) -> DetectIrRecord:
        ir_record = DetectIrRecord(
            filename=labelme_record.imagePath,
            width=labelme_record.imageWidth,
            height=labelme_record.imageHeight
        )
        for labelme_shape in labelme_record.shapes:
            if labelme_shape.shape_type != 'rectangle':
                continue
            ir_object = DetectIrObject(
                label=labelme_shape.label,
                x_min=labelme_shape.points[0][0],
                y_min=labelme_shape.points[0][1],
                x_max=labelme_shape.points[1][0],
                y_max=labelme_shape.points[1][1],
            )
            ir_record.objects.append(ir_object)
        return ir_record
        
    @staticmethod
    def from_ir(ir_record: DetectIrRecord) -> LabelmeRecord:
        labelme_record = LabelmeRecord(
            imagePath=ir_record.filename,
            imageWidth=ir_record.width,
            imageHeight=ir_record.height
        )
        for ir_object in ir_record.objects:
            labelme_shape = LabelmeShape(
                label=ir_object.label,
                shape_type='rectangle',
                points=[[ir_object.x_min, ir_object.y_min], 
                        [ir_object.x_max, ir_object.y_max]]
            )
            labelme_record.shapes.append(labelme_shape)
        return labelme_record
        

@dataclass
class YoloObject:
    label: str
    x_center: float
    y_center: float
    width: float
    height: float
    
    
@dataclass
class YoloRecord:
    filename: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    objects: List[YoloObject] = field(default_factory=list)
    
    
class YoloHandler:
    @staticmethod
    def load(filename, **kwargs) -> YoloRecord:
        assert 'image_filename' in kwargs
        assert 'width' in kwargs and 'height' in kwargs

        records = khandy.load_list(filename)
        yolo_record = YoloRecord(
            filename=kwargs.get('image_filename'),
            width=kwargs.get('width'),
            height=kwargs.get('height'))
        for record in records:
            record_parts = record.split()
            yolo_record.objects.append(YoloObject(
                label=record_parts[0],
                x_center=float(record_parts[1]),
                y_center=float(record_parts[2]),
                width=float(record_parts[3]),
                height=float(record_parts[4]),
            ))
        return yolo_record

    @staticmethod
    def save(filename, yolo_record: YoloRecord):
        records = []
        for object in yolo_record.objects:
            records.append(f'{object.label} {object.x_center} {object.y_center} {object.width} {object.height}')
        if not filename.endswith('.txt'):
            filename = filename + '.txt'
        khandy.save_list(filename, records)

    @staticmethod
    def to_ir(yolo_record: YoloRecord) -> DetectIrRecord:
        ir_record = DetectIrRecord(
            filename=yolo_record.filename,
            width=yolo_record.width,
            height=yolo_record.height
        )
        for yolo_object in yolo_record.objects:
            x_min = (yolo_object.x_center - 0.5 * yolo_object.width) * yolo_record.width
            y_min = (yolo_object.y_center - 0.5 * yolo_object.height) * yolo_record.height
            x_max = (yolo_object.x_center + 0.5 * yolo_object.width) * yolo_record.width
            y_max = (yolo_object.y_center + 0.5 * yolo_object.height) * yolo_record.height
            ir_object = DetectIrObject(
                label=yolo_object.label,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max
            )
            ir_record.objects.append(ir_object)
        return ir_record
        
    @staticmethod
    def from_ir(ir_record: DetectIrRecord) -> YoloRecord:
        yolo_record = YoloRecord(
            filename=ir_record.filename,
            width=ir_record.width,
            height=ir_record.height
        )
        for ir_object in ir_record.objects:
            x_center = (ir_object.x_max + ir_object.x_min) / (2 * ir_record.width)
            y_center = (ir_object.y_max + ir_object.y_min) / (2 * ir_record.height)
            width = abs(ir_object.x_max - ir_object.x_min) / ir_record.width
            height = abs(ir_object.y_max - ir_object.y_min) / ir_record.height
            yolo_object = YoloObject(
                label=ir_object.label,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
            yolo_record.objects.append(yolo_object)
        return yolo_record
    
        
@dataclass
class CocoObject:
    label: str
    x_min: float
    y_min: float
    width: float
    height: float
    
    
@dataclass
class CocoRecord:
    filename: str
    width: int
    height: int
    objects: List[CocoObject] = field(default_factory=list)
    

class CocoHandler:
    @staticmethod
    def load(filename, **kwargs) -> List[CocoRecord]:
        json_data = khandy.load_json(filename)
        
        images = json_data['images']
        annotations = json_data['annotations']
        categories = json_data['categories']
        
        label_map = {}
        for cat_item in categories:
            label_map[cat_item['id']] = cat_item['name']
        
        coco_records = OrderedDict()
        for image_item in images:
            coco_records[image_item['id']] = CocoRecord(
                filename=image_item['file_name'],
                width=image_item['width'],
                height=image_item['height'],
                objects=[])
                
        for annotation_item in annotations:
            coco_object = CocoObject(
                label=label_map[annotation_item['category_id']],
                x_min=annotation_item['bbox'][0],
                y_min=annotation_item['bbox'][1],
                width=annotation_item['bbox'][2],
                height=annotation_item['bbox'][3])
            coco_records[annotation_item['image_id']].objects.append(coco_object)
        return list(coco_records.values())
        
    @staticmethod
    def to_ir(coco_record: CocoRecord) -> DetectIrRecord:
        ir_record = DetectIrRecord(
            filename=coco_record.filename,
            width=coco_record.width,
            height=coco_record.height,
        )
        for coco_object in coco_record.objects:
            ir_object = DetectIrObject(
                label=coco_object.label,
                x_min=coco_object.x_min,
                y_min=coco_object.y_min,
                x_max=coco_object.x_min + coco_object.width,
                y_max=coco_object.y_min + coco_object.height
            )
            ir_record.objects.append(ir_object)
        return ir_record

    @staticmethod
    def from_ir(ir_record: DetectIrRecord) -> CocoRecord:
        coco_record = CocoRecord(
            filename=ir_record.filename,
            width=ir_record.width,
            height=ir_record.height
        )
        for ir_object in ir_record.objects:
            coco_object = CocoObject(
                label=ir_object.label,
                x_min=ir_object.x_min,
                y_min=ir_object.y_min,
                width=ir_object.x_max - ir_object.x_min,
                height=ir_object.y_max - ir_object.y_min
            )
            coco_record.objects.append(coco_object)
        return coco_record
        
        
def load_detect(filename, fmt, **kwargs) -> DetectIrRecord:
    if fmt == 'labelme':
        labelme_record = LabelmeHandler.load(filename, **kwargs)
        ir_record = LabelmeHandler.to_ir(labelme_record)
    elif fmt == 'yolo':
        yolo_record = YoloHandler.load(filename, **kwargs)
        ir_record = YoloHandler.to_ir(yolo_record)
    elif fmt in ('voc', 'pascal', 'pascal_voc'):
        pascal_voc_record = PascalVocHandler.load(filename, **kwargs)
        ir_record = PascalVocHandler.to_ir(pascal_voc_record)
    elif fmt == 'coco':
        coco_records = CocoHandler.load(filename, **kwargs)
        ir_record = [CocoHandler.to_ir(coco_record) for coco_record in coco_records]
    else:
        raise ValueError(f"Unsupported detect label fmt. Got {fmt}")
    return ir_record
    
    
def save_detect(filename, ir_record: DetectIrRecord, out_fmt):
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    if out_fmt == 'labelme':
        labelme_record = LabelmeHandler.from_ir(ir_record)
        LabelmeHandler.save(filename, labelme_record)
    elif out_fmt == 'yolo':
        yolo_record = YoloHandler.from_ir(ir_record)
        YoloHandler.save(filename, yolo_record)
    elif out_fmt in ('voc', 'pascal', 'pascal_voc'):
        pascal_voc_record = PascalVocHandler.from_ir(ir_record)
        PascalVocHandler.save(filename, pascal_voc_record)
    elif out_fmt == 'coco':
        raise ValueError("Unsupported for `coco` now!")
    else:
        raise ValueError(f"Unsupported detect label fmt. Got {out_fmt}")


def _get_format(record):
    if isinstance(record, LabelmeRecord):
        return ('labelme',)
    elif isinstance(record, YoloRecord):
        return ('yolo',)
    elif isinstance(record, PascalVocRecord):
        return ('voc', 'pascal', 'pascal_voc')
    elif isinstance(record, CocoRecord):
        return ('coco',)
    elif isinstance(record, DetectIrRecord):
        return ('ir', 'detect_ir')
    else:
        return ()


def convert_detect(record, out_fmt):
    allowed_fmts = ('labelme', 'yolo', 'voc', 'coco', 'pascal', 'pascal_voc', 'ir', 'detect_ir')
    if out_fmt not in allowed_fmts:
        raise ValueError("Unsupported label format conversions for given out_fmt")
    if out_fmt in _get_format(record):
        return record

    if isinstance(record, LabelmeRecord):
        ir_record = LabelmeHandler.to_ir(record)
    elif isinstance(record, YoloRecord):
        ir_record = YoloHandler.to_ir(record)
    elif isinstance(record, PascalVocRecord):
        ir_record = PascalVocHandler.to_ir(record)
    elif isinstance(record, CocoRecord):
        ir_record = CocoHandler.to_ir(record)
    elif isinstance(record, DetectIrRecord):
        ir_record = record
    else:
        raise TypeError('Unsupported type for record')
        
    if out_fmt in ('ir', 'detect_ir'):
        dst_record = ir_record
    elif out_fmt == 'labelme':
        dst_record = LabelmeHandler.from_ir(ir_record)
    elif out_fmt == 'yolo':
        dst_record = YoloHandler.from_ir(ir_record)
    elif out_fmt in ('voc', 'pascal', 'pascal_voc'):
        dst_record = PascalVocHandler.from_ir(ir_record)
    elif out_fmt == 'coco':
        dst_record = CocoHandler.from_ir(ir_record)
    return dst_record
    

def replace_detect_label(record: DetectIrRecord, label_map, ignore=True):
    dst_record = copy.deepcopy(record)
    dst_objects = []
    for ir_object in dst_record.objects:
        if not ignore:
            if ir_object.label in label_map:
                ir_object.label = label_map[ir_object.label]
            dst_objects.append(ir_object)
        else:
            if ir_object.label in label_map:
                ir_object.label = label_map[ir_object.label]
                dst_objects.append(ir_object)
    dst_record.objects = dst_objects
    return dst_record


def load_coco_class_names(filename):
    json_data = khandy.load_json(filename)
    categories = json_data['categories']
    return [cat_item['name'] for cat_item in categories]

