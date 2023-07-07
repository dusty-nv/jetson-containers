#!/usr/bin/env python3
import os
import sys
import pprint
import argparse
import subprocess

from packages import find_package, package_search_dirs, list_packages
from l4t_version import L4T_VERSION
from base import get_l4t_base


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
    

def build_container(name, packages, base=get_l4t_base(), simulate=False):
    """
    Build container chain of packages
    """
    if isinstance(packages, str):
        packages = [packages]
        
    if len(packages) == 0:
        raise ValueError("must specify at least one package to build")    
            
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
    
    # build the chain of containers
    container_name = name
    
    for idx, package in enumerate(packages):
        # if this isn't the final container in the chain, tag it with the sub-package
        if idx < len(packages) - 1:  
            container_name = f"{name}-{package}"
        
        # build next container
        pkg = find_package(package)
        
        cmd = f"sudo docker build --network=host --tag {container_name} \ \n"
        cmd += f"--file {os.path.join(pkg['path'], pkg['dockerfile'])} \ \n"
        cmd += f"--build-arg BASE_IMAGE={base} \ \n" 
        cmd += ''.join([f"--build-arg {key}={value} \ \n" for key, value in pkg.get('build_args', {}).items()])
        cmd += " . "
        
        print(f"-- Building container {container_name}")
        print(f"\n{cmd}\n")

        if not simulate:
            subprocess.run(cmd.replace('\ \n', ''), shell=True, check=True)  # remove the line breaks that were added for readability

        base = container_name
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
                        
    parser.add_argument('packages', type=str, nargs='*', default=[], help='packages or configs to build')
    
    parser.add_argument('--name', type=str, default='', help='the name of the output container to build')
    parser.add_argument('--base', type=str, default=get_l4t_base(), help='the base container image to use at the beginning of the build chain')
    parser.add_argument('--package-dirs', type=str, nargs='+', default=[], help='additional package search directories')
    parser.add_argument('--list-packages', action='store_true', help='list information about the found packages and exit')
    parser.add_argument('--simulate', action='store_true', help='print out the build commands without actually building the containers')
    
    args = parser.parse_args()
    
    print(args)
    print(f"-- L4T_VERSION={L4T_VERSION}")
    
    # add package search directories from the user
    package_search_dirs(args.package_dirs)
    
    # list packages
    if args.list_packages:
        pprint.pprint(list_packages(scan=True))
        sys.exit(0)
        
    # build container chain
    build_container(args.name, args.packages, args.base, args.simulate)
    