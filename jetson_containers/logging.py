#!/usr/bin/env python3
import os
import datetime

from .packages import _PACKAGE_ROOT


_LOG_DIRS = {}


def log_dir(type='root'):
    """
    Return the path to the logging directory.
    type can be:  root, build, test, run
    """
    return _LOG_DIRS[type]
    

def set_log_dir(path, type='root', create=True):
    """
    Set the path to the logging directory, and create it if needed.
    type can be:  root, build, test, run
    """
    _LOG_DIRS[type] = path
    
    if create:
        os.makedirs(path, exist_ok=True)
        
    if type == 'root':
        set_log_dir(os.path.join(path, 'build'), 'build', create)
        set_log_dir(os.path.join(path, 'test'), 'test', create)
        set_log_dir(os.path.join(path, 'run'), 'run', create)
        
 
# default log dir is:  jetson-containers/logs
set_log_dir(os.path.join(_PACKAGE_ROOT, 'logs', datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))

