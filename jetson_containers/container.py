#!/usr/bin/env python3
import os
import sys
import traceback
import subprocess

from .packages import find_package, find_packages, resolve_dependencies, validate_dict, _PACKAGE_ROOT
from .l4t_version import L4T_VERSION
from .base import get_l4t_base
from .logging import log_dir


_NEWLINE_=" \\\n"  # used when building command strings


def build_container(name, packages, base=get_l4t_base(), build_flags='', simulate=False, skip_tests=False):
    """
    Multi-stage container build that chains together selected packages.
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
    packages = resolve_dependencies(packages)
    print('-- Building containers ', packages)
    
    # make sure all packages can be found before building any
    for package in packages:    
        find_package(package)
            
    # assign default container name and tag if needed
    if len(name) == 0:   
        name = packages[-1]
    elif name.find(':') < 0 and name[-1] == '/':  # they gave a namespace to build under
        name += packages[-1]
        
    if name.find(':') < 0:
        name += f":r{L4T_VERSION}"
    
    for idx, package in enumerate(packages):
        # tag this build stage with the sub-package
        container_name = f"{name}-{package.replace(':','_')}"

        # generate the logging file (without the extension)
        log_file = os.path.join(log_dir('build'), container_name.replace('/','_')).replace(':','_')
        
        # build next intermediate container
        pkg = find_package(package)
        
        if 'dockerfile' in pkg:
            cmd = f"sudo docker build --network=host --tag {container_name}" + _NEWLINE_
            cmd += f"--file {os.path.join(pkg['path'], pkg['dockerfile'])}" + _NEWLINE_
            cmd += f"--build-arg BASE_IMAGE={base}" + _NEWLINE_
            
            if 'build_args' in pkg:
                cmd += ''.join([f"--build-arg {key}=\"{value}\"" + _NEWLINE_ for key, value in pkg['build_args'].items()])
            
            if 'build_flags' in pkg:
                cmd += pkg['build_flags'] + _NEWLINE_
                
            if build_flags:
                cmd += build_flags + _NEWLINE_
                
            cmd += pkg['path'] + _NEWLINE_ #" . "
            cmd += f"2>&1 | tee {log_file + '.txt'}" + "; exit ${PIPESTATUS[0]}"  # non-tee version:  https://stackoverflow.com/a/34604684
            
            print(f"-- Building container {container_name}")
            print(f"\n{cmd}\n")

            with open(log_file + '.sh', 'w') as cmd_file:   # save the build command to a shell script for future reference
                cmd_file.write('#!/usr/bin/env bash\n\n')
                cmd_file.write(cmd + '\n')
                    
            if not simulate:  # remove the line breaks that were added for readability, and set the shell to bash so we can use $PIPESTATUS 
                status = subprocess.run(cmd.replace(_NEWLINE_, ' '), executable='/bin/bash', shell=True, check=True)  
        else:
            tag_container(base, container_name, simulate)
            
        # run tests on the intermediate container
        if package not in skip_tests and 'intermediate' not in skip_tests:
            test_container(container_name, pkg, simulate)
        
        # use this container as the next base
        base = container_name

    # tag the final container
    tag_container(container_name, name, simulate)
    
    # re-run tests on final container
    for package in packages:
        if package not in skip_tests:
            test_container(container_name, package, simulate)
            
    return container_name
    
    
def build_containers(name, packages, base=get_l4t_base(), build_flags='', simulate=False, skip_errors=False, skip_packages=[], skip_tests=[]):
    """
    Build a set of containers independently.
    Returns true if all containers built successfully, or false if there were any errors.
    Building will be halted on the first error encountered, unless skip_errors is set to true.
    TODO add multiprocessing parallel build support for jobs=-1 (use all CPU cores)
    """
    if not packages:  # build everything (for testing)
        packages = sorted(find_packages([]).keys())
    
    packages = find_packages(packages, skip=skip_packages)
    print('-- Building containers', list(packages.keys()))
    
    status = {}

    for package in packages:
        try:
            container_name = build_container(name, package, base, build_flags, simulate, skip_tests) 
        except Exception as error:
            print(error)
            if not skip_errors:
                return False #raise error #sys.exit(os.EX_SOFTWARE)
            status[package] = (container_name, error)
        else:
            status[package] = (container_name, None)
            
    print(f"\n-- Build logs at:  {log_dir('build')}")
    
    for package, (container_name, error) in status.items():
        msg = f"   * {package} ({container_name}) {'FAILED' if error else 'SUCCESS'}"
        if error is not None:
            msg += f"  ({error})"
        print(msg)
        
    for _, error in status.values():
        if error:
            return False
            
    return True
    
    
def tag_container(source, target, simulate=False):
    """
    Tag a container image
    """
    cmd = f"sudo docker tag {source} {target}"
    print(f"-- Tagging container {source} -> {target}")
    print(f"{cmd}\n")
    
    if not simulate:
        subprocess.run(cmd, shell=True, check=True)
        
    
def test_container(name, package, simulate=False):
    """
    Run tests on a container
    """
    package = find_package(package)
    
    if 'test' not in package:
        return True
        
    for test in package['test']:
        test_cmd = test.split(' ')[0]  # test could be a command with arguments - get just the script/executable name
        test_ext = os.path.splitext(test_cmd)[1]
        log_file = os.path.join(log_dir('test'), f"{name.replace('/','_')}_{test_cmd}").replace(':','_')
        
        cmd = "sudo docker run -t --rm --runtime=nvidia --network=host" + _NEWLINE_
        cmd += f"--volume {package['path']}:/test" + _NEWLINE_
        cmd += f"--volume {os.path.join(_PACKAGE_ROOT, 'data')}:/data" + _NEWLINE_
        cmd += f"--workdir /test" + _NEWLINE_
        cmd += name + _NEWLINE_
        
        if test_ext == ".py":
            cmd += f"python3 {test}" + _NEWLINE_
        elif test_ext == ".sh":
            cmd += f"/bin/bash {test}" + _NEWLINE_
        else:
            cmd += f"{test}" + _NEWLINE_
        
        cmd += f"2>&1 | tee {log_file + '.txt'}" + "; exit ${PIPESTATUS[0]}"
                
        print(f"-- Testing container {name} ({package['name']}/{test})")
        print(f"\n{cmd}\n")
        
        with open(log_file + '.sh', 'w') as cmd_file:
            cmd_file.write('#!/usr/bin/env bash\n\n')
            cmd_file.write(cmd + '\n')
            
        if not simulate:  # TODO: return false on errors 
            status = subprocess.run(cmd.replace(_NEWLINE_, ' '), executable='/bin/bash', shell=True, check=True)
            
    return True
    
        