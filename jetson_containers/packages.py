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
_PACKAGE_DIRS = [os.path.join(_PACKAGE_ROOT, 'packages/*')]
_PACKAGE_KEYS = ['alias', 'build_args', 'build_flags', 'category', 'config', 'depends', 
                 'description', 'disabled', 'dockerfile', 'name', 'notes', 'path', 'test']


def package_search_dirs(package_dirs, scan=False):
    """
    Add a list of directories to search for packages under.
    If scan is true, these directories will be scanned for packages.
    """
    global _PACKAGE_DIRS
    
    if isinstance(package_dirs, str) and len(package_dirs) > 0:
        package_dirs = [package_dirs]
        
    for package_dir in package_dirs:
        if len(package_dir) > 0:
            _PACKAGE_DIRS.append(package_dirs)
    
    if scan:
        scan_packages(_PACKAGE_DIRS, rescan=True)
    
    
def scan_packages(package_dirs=_PACKAGE_DIRS, rescan=False):
    """
    Find packages from the list of provided search paths.
    If a path ends in * wildcard, it will be searched recursively.
    This looks for Dockerfiles and config scripts in these directories.
    Returns a dict of package info from this path and sub-paths.
    """
    global _PACKAGES
    global _PACKAGE_SCAN
    
    # skip scanning if it's already done
    if _PACKAGE_SCAN and not rescan:
        return _PACKAGES
        
    # if this is a list of directories, scan each
    if isinstance(package_dirs, list) and len(package_dirs) > 0:
        for path in package_dirs:
            scan_packages(path)
            
        _PACKAGE_SCAN = True  # flag that all dirs have been scanned
        
        for key in _PACKAGES.copy():  # make sure all dependencies are met
            try:
                resolve_dependencies(key)
            except Exception as error:
                print(f"-- Package {key} has missing dependencies, disabling...  ({error})")
                del _PACKAGES[key]
                
        return _PACKAGES
    elif isinstance(package_dirs, str) and len(package_dirs) > 0:
        path = package_dirs
    else:
        raise ValueError(f"package_dirs should be a valid string or list")
        
    # check for wildcard at end of path to scan recursively
    #print(f"-- Searching {path} for packages...")
    
    recursive = (path[-1] == '*')
    path = path.rstrip('*').rstrip('/')
    
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
            
        if os.path.isdir(entry_path) and recursive:
            scan_packages(os.path.join(entry_path, '*'))
        elif os.path.isfile(entry_path):
            if entry.lower().find('dockerfile') >= 0:
                package['dockerfile'] = entry
            elif entry == 'test.py' or entry == 'test.sh':
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
    packages = config_package(package)  # returns a list (including subpackages)

    for pkg in packages:
        _PACKAGES[pkg['name']] = pkg
        
    return _PACKAGES

    
def find_package(package, required=True, scan=True):
    """
    Find a package by name or alias, and return it's configuration dict.
    This filters the names with pattern matching using shell-style wildcards.
    If required is true, an exception will be thrown if the package can't be found.
    """
    if validate_dict(package):
        return package
        
    if scan:
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
        if packages == '*' or packages == 'all' or len(packages) == 0:
            return skip_packages(_PACKAGES, skip) 
        else:
            packages = [packages]
    elif not isinstance(packages, list):
        raise ValueError("packages argument must be a string or a list of strings")
        
    if len(packages) == 0 or not packages[0]:
        return skip_packages(_PACKAGES, skip)
    
    found_packages = {}
    
    for search_pattern in packages:
        found_package = False
        
        for key, pkg in _PACKAGES.items():
            names = [key, pkg['name']] + pkg.get('alias', [])
            
            if len(fnmatch.filter(names, search_pattern)) > 0:
                found_packages[pkg['name']] = pkg
                found_package = True
           
        if required and not found_package:
            raise ValueError(f"couldn't find package:  {search_pattern}")
            
        """
        matches = fnmatch.filter(list(_PACKAGES.keys()), search_pattern)
        
        if required and len(matches) == 0:
            raise ValueError(f"couldn't find package:  {search_pattern}")
            
        for match in matches:
            found_packages[match] = _PACKAGES[match]
        """
        
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
   

def group_packages(packages, key, default=''):
    """
    Group packages by one of their keys, for example 'category' will return a dict
    of all the categories where each category contains the packages belonging to it.
    If a package doesn't have this key, it won't be added unless a default is specified.
    """
    grouped = {}
    
    for name, package in packages.items():
        if key in package:
            value = package[key]
        else:
            if not default:
                continue
            else:
                value = default
                
        grouped.setdefault(value, {})[name] = package
        
    return grouped
    
    
def resolve_dependencies(packages, check=True):
    """
    Recursively expand the list of dependencies to include all sub-dependencies.
    Returns a new list of containers to build which contains all the dependencies.
    """
    if isinstance(packages, str):
        packages = [packages]
    
    # iteratively unroll/expand dependencies until the full list is resolved
    while True:
        packages_org = packages.copy()
        
        for package in packages_org:
            for dependency in find_package(package).get('depends', []):
                package_index = packages.index(package)
                dependency_index = packages.index(dependency) if dependency in packages else -1
                
                if dependency_index < 0:  # dependency not in list, add it before the package
                    packages.insert(package_index, dependency)
                elif dependency_index > package_index:  # dependency after current package, move it to before
                    packages.remove(dependency)
                    packages.insert(package_index, dependency)
      
        if len(packages) == len(packages_org):
            break
     
    # make sure all packages can be found
    if check:
        for package in packages:    
            find_package(package)
        
    return packages


def dependant_packages(package):
    """
    Find the list of all packages that depend on this package.
    """
    if isinstance(package, str):
        package = find_package(package)
        
    dependants = []
    
    for key, pkg in _PACKAGES.items():
        if package == pkg:
            continue
            
        depends = resolve_dependencies(key, check=False)
        
        for dependency in depends:
            if package == find_package(dependency):
                dependants.append(key)
    
    return dependants
    
    
def apply_config(package, config):
    """
    Apply a config dict to an existing package configuration
    """
    if config is None or not isinstance(config, dict):
        return
    
    if validate_dict(config):  # the package config entries are in the top-level dict
        package.update(validate_lists(config))
    elif len(config) == 1:  # nested dict with just one package (merge with existing package)
        name = list(config.keys())[0]
        package['name'] = name
        package.update(validate_lists(config[name]))
    else:
        for pkg_name, pkg in config.items():  # nested dict with multiple subpackages
            for key in _PACKAGE_KEYS:  # apply inherited package info
                if key in package:
                    #print(f"-- Setting {pkg_name} key {key} from {package['name']} to ", package[key])
                    pkg.setdefault(key, package[key])
            package[pkg_name] = validate_lists(pkg)
    
                    
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
            if package is None:  # package was disabled in config script
                return []
                
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
        packages = package  # TODO what if these contain subpackages?
       
    for pkg in packages.copy():  # check to see if any packages were disabled
        if pkg.get('disabled', False):
            print(f"-- Package {pkg['name']} was disabled by its config")
            packages.remove(pkg)
        else:    
            validate_lists(pkg)  # make sure certain entries are lists
        
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
    

def validate_lists(package):
    """
    Make sure that certain config entries are lists.
    """
    def str2list(pkg, key):
        if key in pkg and isinstance(pkg[key], str):
            pkg[key] = [pkg[key]]
                  
    str2list(package, 'alias')
    str2list(package, 'config')
    str2list(package, 'depends')
    str2list(package, 'test')
        
    return package
     
     
def parse_yaml_header(dockerfile):
    """
    Parse YAML configuration from the Dockerfile header
    """
    try:
        txt = ""
        
        with open(dockerfile, 'r') as file:
            in_yaml = False
            while True:
                line = file.readline()
                if len(line) == 0:
                    break
                if line[0] != '#':
                    break
                
                if in_yaml:
                    if line.startswith('#---'):
                        in_yaml = False
                    else:
                        txt += line[1:]
                elif line.startswith('#---'):
                        in_yaml = True
                
        if len(txt) == 0:
            return None

        config = yaml.safe_load(txt)

        if validate_dict(config):
            return config
        else:
            print(f"-- YAML header from {dockerfile} contained unknown/invalid entries, ignoring...")
            print(txt)
            
    except Exception as error:
        print(f"-- Error parsing YAML from {dockerfile}:  {error}")
        
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
        