#!/usr/bin/env python3
from .args import *
from .audio import *
from .image import *
from .log import *
from .model import *
from .prompts import *
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


def get_class_that_defined_method(meth):
    """
    Given a function or method, return the class type it belongs to
    https://stackoverflow.com/a/25959545
    """
    import inspect
    
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = meth.__func__  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                      None)
        if isinstance(cls, type):
            return cls
    return None  # not required since None would have been implicitly returned anyway