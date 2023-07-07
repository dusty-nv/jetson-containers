#!/usr/bin/env python3
import os
import sys
import importlib


# this takes care of 'import jetson_containers' from sub-packages
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

_PACKAGES = {}
_PACKAGE_DIRS = [os.path.join(os.path.dirname(os.path.dirname(__file__)), 'packages')]
 
    
def package_search_dirs(package_dirs, scan=False):
    """
    Add a list of directories to search for packages under.
    If scan is true, these directories will be scanned for packages.
    """
    global _PACKAGE_DIRS
    
    if isinstance(package_dirs, str):
        package_dirs = [package_dirs]
        
    _PACKAGE_DIRS.extend(package_dirs)
    
    '''
    for _PACKAGE_DIR in _PACKAGE_DIRS:
        _PACKAGE_DIR = os.path.dirname(_PACKAGE_DIR)
        if _PACKAGE_DIR not in sys.path:
            sys.path.append(_PACKAGE_DIR)
    '''
    
    if scan:
        scan_packages(_PACKAGE_DIRS)
    
    
def scan_packages(package_dirs=_PACKAGE_DIRS):
    """
    Recursively find packages in and under the provided search paths.
    This looks for Dockerfiles and config scripts in these directories.
    Returns a dict of package info from this path and sub-paths.
    """
    global _PACKAGES
    
    if isinstance(package_dirs, list):
        for path in package_dirs:
            scan_packages(path)
        return _PACKAGES
    elif isinstance(package_dirs, str):
        path = package_dirs
    else:
        raise ValueError(f"package_dirs should be a string or list")
        
    #print(f"searching '{path}' for packages...")
    
    if not os.path.isdir(path):
        print(f"warning -- package dir '{path}' doesn't exist, skipping...")
        return _PACKAGES
        
    # search this directory for dockerfiles and config scripts
    entries = os.listdir(path)
    package = {'path': path}
    
    for entry in entries:
        entry_path = os.path.join(path, entry)
        
        if entry.startswith('__'):  # skip hidden directories
            continue
            
        if os.path.isdir(entry_path):
            scan_packages(entry_path)
        elif os.path.isfile(entry_path):
            if entry.lower().find('dockerfile') >= 0:
                package['dockerfile'] = entry
            elif entry == 'config.py':
                package['config'] = entry
            elif entry == 'test.py':
                package['test'] = entry
                
    # skip directories with no dockerfiles
    if 'dockerfile' not in package: #len(package) > 0:
        #print(f"warning -- didn't find a Dockerfile under '{path}', skipping...")
        return _PACKAGES
        
    # configure new packages
    package_name = os.path.basename(path)
    
    if package_name in _PACKAGES:
        return _PACKAGES
        
    package['name'] = package_name
    package = config_package(package)
    
    for pkg in package:
        _PACKAGES[pkg['name']] = pkg
        
    return _PACKAGES

    
def find_package(package, raise_exception=True):
    """
    Find a package by name or alias, and return it's configuration dict.
    An exception will be thrown if the package can't be found and raise_exception is true.
    """
    scan_packages()
    
    for key, pkg in _PACKAGES.items():
        if package == key or package in pkg.get('alias', []):
            return pkg
         
    if raise_exception:
        raise ValueError(f"couldn't find package:  {package}")
    else:
        return None
        

def list_packages(scan=False):
    """
    Return the dictionary of found packages.
    If scan is true, the package directories will be searched.
    """
    if scan:
        scan_packages()
        
    return _PACKAGES

    
def config_package(package):
    """
    Run a package's config.py script if it has one
    """
    if isinstance(package, str):
        package = find_package(package)
    elif not isinstance(package, dict):
        raise ValueError("package should either be a string or dict")
        
    if 'config' not in package:
        return validate_package(package)
        
    config_path = os.path.join(package['path'], package['config'])
    print(f"-- Importing {config_path}")
    
    # import the config script
    module_name = f"packages.{package['name']}.config"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    module.package = package   # add current package's dict as a global
    spec.loader.exec_module(module)
    package = module.package
        
    return validate_package(package)
    

def validate_package(package):
    """
    Validate/check a package's configuration
    """
    packages = []
    
    if isinstance(package, dict):
        for key, value in package.items():  # check for sub-packages
            if isinstance(value, dict) and 'dockerfile' in value:
                value['name'] = key  # assuming name based on key
                packages.append(value)
        if len(packages) == 0:  # there were no sub-packages
            packages.append(package)
    elif isinstance(package, list):
        packages = package
       
    # make sure certain entries are lists
    def str2list(pkg, key):
        if key in pkg and isinstance(pkg[key], str):
            pkg[key] = [pkg[key]]
                  
    for pkg in packages:
        str2list(pkg, 'alias')
        str2list(pkg, 'depends')
    
    return packages
    
        
'''
#_CURRENT_PACKAGE = None

def package_alias(alias):
    """
    Add an alternate name (or list of names) to the current package.
    It will then be able to be found using these other aliases.
    """
    if isinstance(alias, str):
        alias = [alias]
    elif not isinstance(alias, list):
        raise ValueError("alias should be a string or list of strings")
        
    package_config({'alias': alias})
    
    
def package_build_args(env):
    """
    Add docker --build-arg options to the current package
    env should be a dict where the keys are the ARG name
    """
    build_args = ""
    
    for key, value in env.items():
        build_args += f"--build-arg {key}={value} "
    
    package_config({'build_args': build_args})

    
def package_config(config):
    """
    Apply a dict to the current package's configuration
    """
    global _CURRENT_PACKAGE
    
    print('package_config dir()', dir())
    print('CURRENT_PACKAGE', _CURRENT_PACKAGE)
    
    if _CURRENT_PACKAGE is None:
        raise ValueError("a package isn't currently being configured")
        
    _CURRENT_PACKAGE.update(config)
    
    
def package_depends(package):
    """
    Apply a build dependency to the current package.
    """
    if isinstance(package, str):
        package = [package]
    elif not isinstance(package, list):
        raise ValueError("package should be a string or list of strings")
        
    package_config({'depends': package})
    

def package_name(name):
    """
    Set the current package's name.
    """
    package_config({'name': name})
'''  
        