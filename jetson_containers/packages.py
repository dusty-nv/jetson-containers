#!/usr/bin/env python3
import os
import sys
import json
import fnmatch
import importlib

try:
    import yaml
except Exception as error:
    print(error)
    print("Failed to import yaml - please install in it either of these ways:")
    print("$ sudo apt-get update && sudo apt-get install python3-yaml")
    print("$ pip3 install pyyaml")
    sys.exit(os.EX_UNAVAILABLE)
    
# package globals
_PACKAGES = {}

_PACKAGE_SCAN = False
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(__file__))
_PACKAGE_DIRS = [os.path.join(_PACKAGE_ROOT, 'packages'), os.path.join(_PACKAGE_ROOT, 'config')]
_PACKAGE_KEYS = ['alias', 'build_args', 'build_flags', 'config', 'depends', 'description', 'dockerfile', 'name', 'path', 'test']


def package_search_dirs(package_dirs, scan=False):
    """
    Add a list of directories to search for packages under.
    If scan is true, these directories will be scanned for packages.
    """
    global _PACKAGE_DIRS
    
    if isinstance(package_dirs, str) and len(package_dirs) > 0:
        package_dirs = [package_dirs]
        
    _PACKAGE_DIRS.extend(package_dirs)
    
    if scan:
        scan_packages(_PACKAGE_DIRS, rescan=True)
    
    
def scan_packages(package_dirs=_PACKAGE_DIRS, rescan=False):
    """
    Recursively find packages in and under the provided search paths.
    This looks for Dockerfiles and config scripts in these directories.
    Returns a dict of package info from this path and sub-paths.
    """
    global _PACKAGES
    global _PACKAGE_SCAN
    
    # skip scanning if it's already done
    if _PACKAGE_SCAN and not rescan:
        return _PACKAGES
        
    # if this is a list of directories, scan each
    if isinstance(package_dirs, list):
        for path in package_dirs:
            scan_packages(path)
        _PACKAGE_SCAN = True
        return _PACKAGES
    elif isinstance(package_dirs, str):
        path = package_dirs
    else:
        raise ValueError(f"package_dirs should be a string or list")
        
    # scan this specific directory for packages
    #print(f"-- Searching {path} for packages...")
    
    if not os.path.isdir(path):
        print(f"-- Package dir '{path}' doesn't exist, skipping...")
        return _PACKAGES
        
    # search this directory for dockerfiles and config scripts
    entries = os.listdir(path)
    package = {'path': path, 'config': [], 'test': []}
    
    for entry in entries:
        entry_path = os.path.join(path, entry)
        
        if not entry or entry.startswith('__'):  # skip hidden directories
            continue
            
        if os.path.isdir(entry_path):
            scan_packages(entry_path)
        elif os.path.isfile(entry_path):
            if entry.lower().find('dockerfile') >= 0:
                package['dockerfile'] = entry
            elif entry == 'test.py':
                package['test'].append(entry)
            elif entry == 'config.py':
                package['config'].append(entry)
            elif validate_config(entry_path):
                package['config'].append(entry)
                
    # skip directories with no dockerfiles or configuration
    if 'dockerfile' not in package and len(package['config']) == 0:
        #print(f"-- Skipping '{path}' (didn't find a Dockerfile or package config)")
        return _PACKAGES
        
    # configure new packages
    package_name = os.path.basename(path)
    
    if package_name in _PACKAGES:
        return _PACKAGES
        
    package['name'] = package_name
    package = config_package(package)  # returns a list (including subpackages)

    for pkg in package:
        _PACKAGES[pkg['name']] = pkg
        
    return _PACKAGES

    
def find_package(package, required=True, scan=True):
    """
    Find a package by name or alias, and return it's configuration dict.
    This filters the names with pattern matching using shell-style wildcards.
    If required is true, an exception will be thrown if the package can't be found.
    """
    scan_packages()

    for key, pkg in _PACKAGES.items():
        names = [key, pkg['name']] + pkg.get('alias', [])

        if len(fnmatch.filter(names, package)) > 0:
            return pkg
        
    if required:
        raise ValueError(f"couldn't find package:  {package}")
    else:
        return None
        
        
def find_packages(packages, required=True, scan=True, skip=[]):
    """
    Find a set of packages by name or alias, and return them in a dict.
    This filters the names with pattern matching using shell-style wildcards.
    If required is true, an exception will be thrown if any of the packages can't be found.
    """
    if scan:
        scan_packages()
    
    if isinstance(packages, str):
        packages = [packages]
        
    if len(packages) == 0:
        return skip_packages(_PACKAGES, skip)
    
    found_packages = {}
    
    for search_pattern in packages:
        matches = fnmatch.filter(list(_PACKAGES.keys()), search_pattern)
        
        if required and len(matches) == 0:
            raise ValueError(f"couldn't find package:  {search_pattern}")
            
        for match in matches:
            found_packages[match] = _PACKAGES[match]
            
    return skip_packages(found_packages, skip)
    

def skip_packages(packages, skip):
    """
    Filter a dict of packages by a list of names to skip (can use wildcards)
    """
    if isinstance(skip, str):
        skip = [skip]
        
    if len(skip) == 0:
        return packages
        
    filtered = {}
    
    for key, value in packages.items():
        found_match = False
        for skip_pattern in skip:
            if fnmatch.fnmatch(key, skip_pattern):
                found_match = True
        if not found_match:
            filtered[key] = value
                
    return filtered
    
   
def apply_config(package, config):
    """
    Apply a config dict to an existing package configuration
    """
    if config is None or not isinstance(config, dict):
        return
    
    if validate_dict(config):  # the package config entries are in the top-level dict
        package.update(config)
    elif len(config) == 1:  # nested dict with just one package (merge with existing package)
        name = list(config.keys())[0]
        package['name'] = name
        package.update(config[name])
    else:
        for pkg_name, pkg in config.items():  # nested dict with multiple subpackages
            for key in _PACKAGE_KEYS:  # apply inherited package info
                print(f"-- Setting {pkg_name} key {key} from {package[name]} to ", package[key])
                pkg.setdefault(key, package[key])
            package[pkg_name] = pkg
    
                    
def config_package(package):
    """
    Run a package's config.py or JSON if it has one
    """
    if isinstance(package, str):
        package = find_package(package)
    elif not isinstance(package, dict):
        raise ValueError("package should either be a string or dict")
               
    if 'dockerfile' in package:
        config = parse_yaml_header(os.path.join(package['path'], package['dockerfile']))
        apply_config(package, config)
        
    if len(package['config']) == 0:
        return validate_package(package)
        
    for config_filename in package['config']:
        config_ext = os.path.splitext(config_filename)[1]
        config_path = os.path.join(package['path'], config_filename)
        
        if config_ext == '.py':
            print(f"-- Loading {config_path}")
            module_name = f"packages.{package['name']}.config"
            spec = importlib.util.spec_from_file_location(module_name, config_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            module.package = package   # add current package's dict as a global
            spec.loader.exec_module(module)
            package = module.package
            
        elif config_ext == '.json' or config_ext == '.yaml':
            print(f"-- Loading {config_path}")
            config = validate_config(config_path)  # load and validate the config file
            apply_config(package, config)
                    
    return validate_package(package)
    

def validate_package(package):
    """
    Validate/check a package's configuration, returning a list (i.e. of subpackages)
    """
    packages = []
    
    if isinstance(package, dict):
        for key, value in package.items():  # check for sub-packages
            if validate_dict(value):
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
    

def validate_config(path):
    """
    Return a well-formed package configuration JSON or YAML file, or None on error.
    """
    ext = os.path.splitext(path)[1]
    
    if ext != '.json' and ext != '.yaml':
        return None
        
    try:
        with open(path, 'r') as file:
            if ext == '.json':
                config = json.load(file)
            elif ext == '.yaml':
                config = yaml.safe_load(file)
    except Exception as err:
        print(f"-- Error loading {path}")
        print(err)
        return None

    if not isinstance(config, dict) or len(config) == 0:
        return None
    
    if not validate_dict(config):  # see if the top-level dict contains the package configuration entries themselves
        for package_name, package in config.items():  # see if this is a nested dict with one or multiple subpackages
            if not validate_dict(package):
                return None
        
    return config

    
def validate_dict(package):
    """
    Return true if this is a package configuration dict.
    """
    if not isinstance(package, dict):
        return False
        
    for key, value in package.items():
        if key not in _PACKAGE_KEYS:
            #print(f"-- Unknown key '{key}' in package config:", value)
            return False
            
    return True
    

def parse_yaml_header(dockerfile):
    """
    Parse YAML configuration from the Dockerfile header
    """
    try:
        txt = ""
        
        with open(dockerfile, 'r') as file:
            while True:
                line = file.readline()
                if len(line) == 0:
                    break
                if line[0] != '#':
                    break
                txt += line[1:]
                
        if len(txt) == 0:
            return None

        config = yaml.safe_load(txt)

        if validate_dict(config):
            return config
            
    except Exception as error:
        print(f"Error parsing YAML from {dockerfile}:  {error}")
        
    return None

    
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
        