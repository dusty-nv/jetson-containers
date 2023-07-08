#!/usr/bin/env python3
import os
import subprocess

from .packages import find_package, validate_dict
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
    Build container chain of packages
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
            cmd += f"2>&1 | tee {log_file + '.txt'}"
            
            print(f"-- Building container {container_name}")
            print(f"\n{cmd}\n")

        if not simulate:
            with open(log_file + '.sh', 'w') as cmd_file:   # save the build command to a shell script for future reference
                cmd_file.write('#!/usr/bin/env bash\n\n')
                cmd_file.write(cmd + '\n')
            
            subprocess.run(cmd.replace('\ \n', ''), shell=True, check=True)  # remove the line breaks that were added for readability

        base = container_name

    # tag the final container
    cmd = f"sudo docker tag {container_name} {name}"
    print(f"-- Tagging container {container_name} -> {name}")
    print(f"{cmd}\n")
    
    if not simulate:
        subprocess.run(cmd, shell=True, check=True)
    