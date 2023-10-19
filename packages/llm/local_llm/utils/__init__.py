#!/usr/bin/env python3
from .image import *
from .log import *
from .model import *
from .prompt import *
from .table import *
from .tensor import *


def replace_text(text, dict):
    """
    Replace instances of each of the keys in dict in the text string with the values in dict
    """
    for key, value in dict.items():
        text = text.replace(key, value)
    return text    
    
    
class AttributeDict(dict):
    """
    A dict where keys are available as attributes
    https://stackoverflow.com/a/14620633
    """
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
