#!/usr/bin/env python3
import os
import sys
import traceback
import subprocess

from .packages import find_package, find_packages, validate_dict
from .l4t_version import L4T_VERSION
from .base import get_l4t_base
from .logging import log_dir


def unroll_dependencies(packages):
    """
    Expand the dependencies in the list of containers to build
    """
    if isinstance(packages, str):
        packages = [packages]
    
    while True:
        packages_org = packages.copy()
        
        for package in packages_org:
            for dependency in find_package(package).get('depends', []):
                if dependency not in packages:
                    packages.insert(packages.index(package), dependency)
                
        if len(packages) == len(packages_org):
            break
            
    return packages
    

def build_container(name, packages, base=get_l4t_base(), build_flags='', simulate=False):
    """
    Multi-stage container build of that chains together selected packages.
    """
    if isinstance(packages, str):
        packages = [packages]
    elif validate_dict(packages):
        packages = [packages['name']]
        
    if len(packages) == 0:
        raise ValueError("must specify at least one package to build")    
           
    if not base:
        base = get_l4t_base()
        
    # add all dependencies to the build tree
    packages = unroll_dependencies(packages)
    print('-- Building containers ', packages)
    
    # make sure all packages can be found before building any
    for package in packages:    
        find_package(package)
            
    # assign default container name and tag if needed
    if len(name) == 0:   
        name = packages[-1]
          
    if name.find(':') < 0:
        name += f":r{L4T_VERSION}"
    
    for idx, package in enumerate(packages):
        # tag this build stage with the sub-package
        container_name = f"{name}-{package}"

        # generate the logging file (without the extension)
        log_file = os.path.join(log_dir('build'), container_name).replace(':','_')
        
        # build next container
        pkg = find_package(package)
        
        if 'dockerfile' in pkg:
            cmd = f"sudo docker build --network=host --tag {container_name} \ \n"
            cmd += f"--file {os.path.join(pkg['path'], pkg['dockerfile'])} \ \n"
            cmd += f"--build-arg BASE_IMAGE={base} \ \n" 
            
            if 'build_args' in pkg:
                cmd += ''.join([f"--build-arg {key}=\"{value}\" \ \n" for key, value in pkg['build_args'].items()])
            
            if 'build_flags' in pkg:
                cmd += pkg['build_flags'] + ' \ \n'
                
            if build_flags:
                cmd += build_flags + ' \ \n'
                
            cmd += pkg['path'] + ' \ \n' #" . "
            cmd += f"2>&1 | tee {log_file + '.txt'}" + "; exit ${PIPESTATUS[0]}"  # non-tee version:  https://stackoverflow.com/a/34604684
            
            print(f"-- Building container {container_name}")
            print(f"\n{cmd}\n")

        if not simulate:
            with open(log_file + '.sh', 'w') as cmd_file:   # save the build command to a shell script for future reference
                cmd_file.write('#!/usr/bin/env bash\n\n')
                cmd_file.write(cmd + '\n')
            
            # remove the line breaks that were added for readability, and set the shell to bash so we can use $PIPESTATUS 
            status = subprocess.run(cmd.replace('\ \n', ''), executable='/bin/bash', shell=True, check=True)  

        base = container_name

    # tag the final container
    cmd = f"sudo docker tag {container_name} {name}"
    print(f"-- Tagging container {container_name} -> {name}")
    print(f"{cmd}\n")
    
    if not simulate:
        subprocess.run(cmd, shell=True, check=True)
    
    
def build_containers(name, packages, base=get_l4t_base(), build_flags='', simulate=False, skip_errors=False, skip_packages=[]):
    """
    Build a set of containers independently.
    TODO add support for jobs=-1 (use all CPU cores)
    TODO add return False on error
    """
    if not packages:  # build everything (for testing)
        packages = sorted(find_packages([]).keys())
    
    packages = find_packages(packages, skip=skip_packages)
    print('-- Building containers', list(packages.keys()))
    
    status = {}

    for package in packages:
        try:
            build_container(name, package, base, build_flags, simulate) 
        except Exception as error:
            print(error)
            if not skip_errors:
                sys.exit(os.EX_SOFTWARE)
            status[package] = (False, error)
        else:
            status[package] = (True, None)
            
    print(f"\n-- Build logs at:  {log_dir('build')}")
    
    for package, (success, error) in status.items():
        msg = f"   * {package} {'SUCCESS' if success else 'FAILED'}"
        if error is not None:
            msg += f"  ({error})"
        print(msg)
        
    for success, _ in status.values():
        if not success:
            return False
            
    return True
    
        