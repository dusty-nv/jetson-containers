#!/usr/bin/env python3
from .args import *
from .audio import *
from .image import *
from .keyboard import *
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
    A dict where keys are available as attributes:
    
      https://stackoverflow.com/a/14620633
      
    So you can do things like:
    
      x = AttributeDict(a=1, b=2, c=3)
      x.d = x.c - x['b']
      x['e'] = 'abc'
      
    This is using the __getattr__ / __setattr__ implementation
    (as opposed to the more concise original commented out below)
    because of memory leaks encountered without it:
    
      https://bugs.python.org/issue1469629
      
    TODO - rename this to ConfigDict or NamedDict?
    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __getstate__(self):
        return self.__dict__


'''    
class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
'''

      
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
    
    
def ends_with_token(input, tokens, tokenizer=None):
    """
    Check to see if the list of input tokens ends with any of the list of stop tokens.
    This is typically used to check if the model produces a stop token like </s> or <eos>
    """
    if not isinstance(input, list):
        input = [input]
        
    if not isinstance(tokens, list):
        tokens = [tokens]
     
    if len(input) == 0 or len(tokens) == 0:
        return False
        
    for stop_token in tokens:
        if isinstance(stop_token, list):
            if len(stop_token) == 1:
                if input[-1] == stop_token[0]:
                    return True
            elif len(input) >= len(stop_token):
                if tokenizer:
                    input_text = tokenizer.decode(input, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    stop_text = tokenizer.decode(stop_token, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    #print('input_text', input_text, 'stop_text', f"'{stop_text}'")
                    if input_text.endswith(stop_text):
                        #print('STOPPING TEXT')
                        return True
                else:
                    if input[-len(stop_token):] == stop_token:
                        return True
        elif input[-1] == stop_token:
            return True
            
    return False
    
    
def wrap_text(font, image, text='', x=5, y=5, **kwargs):
    """"
    Utility for cudaFont that draws text on a image with word wrapping.
    Returns the new y-coordinate after the text wrapping was applied.
    """
    text_color=kwargs.get("color", font.White) 
    background_color=kwargs.get("background", font.Gray40)
    line_spacing = kwargs.get("line_spacing", 38)
    line_length = kwargs.get("line_length", image.width // 16)

    text = text.split()
    current_line = ""

    for n, word in enumerate(text):
        if len(current_line) + len(word) <= line_length:
            current_line = current_line + word + " "
            
            if n == len(text) - 1:
                font.OverlayText(image, text=current_line, x=x, y=y, color=text_color, background=background_color)
                return y + line_spacing
        else:
            current_line = current_line.strip()
            font.OverlayText(image, text=current_line, x=x, y=y, color=text_color, background=background_color)
            current_line = word + " "
            y=y+line_spacing
    return y
    