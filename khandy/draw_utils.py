import numpy as np
import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageColor


def _is_legal_color(color):
    if color is None:
        return True
    if isinstance(color, str):
        return True
    return isinstance(color, (tuple, list)) and len(color) == 3
        

def _normalize_color(color, pil_mode, swap_rgb=False):
    if color is None:
        return color
    if isinstance(color, str):
        color = ImageColor.getrgb(color)
    gray = color[0]
    if swap_rgb:
        color = (color[2], color[1], color[0])
    if pil_mode == 'L':
        color = gray
    return color
    
    
def draw_text(image, text, position, color=(255,0,0), font=None, font_size=15):
    """Draws text on given image.
    
    Args:
        image (ndarray).
        text (str): text to be drawn.
        position (Tuple[int, int]): position where to be drawn.
        color (List[Union[str, Tuple[int, int, int]]]): text color.
        font (str):  A filename or file-like object containing a TrueType font. If the file is not found in this 
            filename, the loader may also search in other directories, such as the `fonts/` directory on Windows
            or `/Library/Fonts/`, `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.

    References:
        torchvision.utils.draw_bounding_boxes
    """
    if isinstance(image, np.ndarray):
        # PIL.Image.fromarray fails with uint16 arrays
        # https://github.com/python-pillow/Pillow/issues/1514
        if (image.dtype == np.uint16) and (image.ndim != 2):
            image = (image / 256).astype(np.uint8)
        pil_image = Image.fromarray(image)
    elif isinstance(image, PIL.Image.Image):
        pil_image = image
    else:
        raise TypeError('Unsupported image type!')
    assert pil_image.mode in ['L', 'RGB', 'RGBA']
    
    assert _is_legal_color(color)
    color = _normalize_color(color, pil_image.mode, isinstance(image, np.ndarray))
    
    if font is None:
        font_object = ImageFont.load_default()
    else:
        font_object = ImageFont.truetype(font, size=font_size)
    
    draw = ImageDraw.Draw(pil_image)
    draw.text((position[0], position[1]), text, 
              fill=color, font=font_object)

    if isinstance(image, np.ndarray):
        return np.asarray(pil_image)
    return pil_image


def draw_bounding_boxes(image, boxes, labels=None, colors=None,
                        fill=False, width=1, font=None, font_size=15):
    """Draws bounding boxes on given image.

    Args:
        image (ndarray).
        boxes (ndarray): ndarray of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format.
        labels (List[str]): List containing the labels of bounding boxes.
        colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of bounding boxes or labels.
        fill (bool): If `True` fills the bounding box with specified color.
        width (int): Width of bounding box.
        font (str):  A filename or file-like object containing a TrueType font. If the file is not found in this 
            filename, the loader may also search in other directories, such as the `fonts/` directory on Windows
            or `/Library/Fonts/`, `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.

    References:
        torchvision.utils.draw_bounding_boxes
    """
    if isinstance(image, np.ndarray):
        # PIL.Image.fromarray fails with uint16 arrays
        # https://github.com/python-pillow/Pillow/issues/1514
        if (image.dtype == np.uint16) and (image.ndim != 2):
            image = (image / 256).astype(np.uint8)
        pil_image = Image.fromarray(image)
    elif isinstance(image, PIL.Image.Image):
        pil_image = image
    else:
        raise TypeError('Unsupported image type!')
    pil_image = pil_image.convert('RGB')
    
    if font is None:
        font_object = ImageFont.load_default()
    else:
        font_object = ImageFont.truetype(font, size=font_size)

    if fill:
        draw = ImageDraw.Draw(pil_image, "RGBA")
    else:
        draw = ImageDraw.Draw(pil_image)

    for i, bbox in enumerate(boxes):
        if colors is None:
            color = None
        else:
            color = colors[i]
            
        assert _is_legal_color(color)
        color = _normalize_color(color, pil_image.mode, isinstance(image, np.ndarray))
        
        if fill:
            if color is None:
                fill_color = (255, 255, 255, 100)
            elif isinstance(color, str):
                # This will automatically raise Error if rgb cannot be parsed.
                fill_color = ImageColor.getrgb(color) + (100,)
            elif isinstance(color, tuple):
                fill_color = color + (100,)
            # the first argument of ImageDraw.rectangle:
            # in old version only supports [(x0, y0), (x1, y1)]
            # in new version supports either [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]
            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], width=width, outline=color)

        if labels is not None:
            margin = width + 1
            draw.text((bbox[0] + margin, bbox[1] + margin), labels[i], fill=color, font=font_object)

    if isinstance(image, np.ndarray):
        return np.asarray(pil_image)
    return pil_image
    
    
