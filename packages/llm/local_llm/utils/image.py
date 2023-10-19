#!/usr/bin/env python3
import logging

from PIL import Image
from io import BytesIO


ImageExtensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')


def load_image(path):
    """
    Load an image from a local path or URL
    """
    if path.startswith('http') or path.startswith('https'):
        logging.debug(f'-- downloading {path}')
        response = requests.get(path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        logging.debug(f'-- loading {path}')
        image = Image.open(path).convert('RGB')
        
    return image