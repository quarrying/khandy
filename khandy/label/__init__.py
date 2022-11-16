from .detect import *


def _get_format(record):
    if isinstance(record, LabelmeRecord):
        return ('labelme',)
    elif isinstance(record, YoloRecord):
        return ('yolo',)
    elif isinstance(record, PascalVocRecord):
        return ('voc', 'pascal', 'pascal_voc')
    elif isinstance(record, CocoRecord):
        return ('coco',)
    else:
        return ()
        
    
def load(filename, fmt, **kwargs):
    if fmt == 'labelme':
        record = LabelmeHandler.load(filename)
    elif fmt == 'yolo':
        record = YoloHandler.load(filename)
    elif fmt in ('voc', 'pascal', 'pascal_voc'):
        record = PascalVocHandler.load(filename)
    elif fmt == 'coco':
        record = CocoDetectHandler.load(filename, **kwargs)
    else:
        raise ValueError(f"Unsupported detect label fmt. Got {fmt}")
    return record
    
    
def save(filename, record):
    if isinstance(record, LabelmeRecord):
        LabelmeHandler.save(filename, record)
    elif isinstance(record, YoloRecord):
        YoloHandler.save(filename, record)
    elif isinstance(record, PascalVocRecord):
        PascalVocHandler.save(filename, record)
    elif isinstance(record, CocoRecord):
        raise ValueError("Unsupported for CocoRecord now!")
    else:
        raise ValueError("Unsupported type!")
        
        
def convert(record, out_fmt):
    allowed_fmts = ("labelme", "yolo", "voc", "coco", 'pascal', 'pascal_voc')
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
        ir_record = CocoDetectHandler.to_ir(record)
    else:
        raise ValueError('Unsupported type for record')
        
    if out_fmt == 'labelme':
        dst_record = LabelmeHandler.from_ir(ir_record)
    elif out_fmt == 'yolo':
        dst_record = YoloHandler.from_ir(ir_record)
    elif out_fmt in ('voc', 'pascal', 'pascal_voc'):
        dst_record = PascalVocHandler.from_ir(ir_record)
    elif out_fmt == 'coco':
        dst_record = CocoDetectHandler.from_ir(ir_record)
    return dst_record
    
