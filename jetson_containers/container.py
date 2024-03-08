#!/usr/bin/env python3
import os
import sys
import copy
import time
import json
import pprint
import fnmatch
import logging
import traceback
import subprocess
import dockerhub_api 

from .packages import find_package, find_packages, resolve_dependencies, validate_dict, _PACKAGE_ROOT
from .l4t_version import L4T_VERSION, l4t_version_from_tag, l4t_version_compatible, get_l4t_base
from .utils import split_container_name, query_yes_no, needs_sudo, sudo_prefix
from .logging import log_dir, log_debug, pprint_debug

from packaging.version import Version


_NEWLINE_=" \\\n"  # used when building command strings


def build_container(name, packages, base=get_l4t_base(), build_flags='', simulate=False, skip_tests=[], test_only=[], push='', no_github_api=False):
    """
    Multi-stage container build that chains together selected packages into one container image.
    For example, `['pytorch', 'tensorflow']` would build a container that had both pytorch and tensorflow in it.
    
    Parameters:
      name (str) -- name of container image to build (or a namespace to build under, ending in /)
                    if empty, a default name will be assigned based on the package(s) selected.           
      packages (list[str]) -- list of package names to build (into one container)
      base (str) -- base container image to use (defaults to l4t-base or l4t-jetpack)
      build_flags (str) -- arguments to add to the 'docker build' command
      simulate (bool) -- if true, just print out the commands that would have been run
      skip_tests (list[str]) -- list of tests to skip (or 'all' or 'intermediate')
      test_only (list[str]) -- only test these specified packages, skipping all other tests
      push (str) -- name of repository or user to push container to (no push if blank)
      no_github_api (bool) -- if true, use custom Dockerfile with no `ADD https://api.github.com/repos/...` line.
      
    Returns: 
      The full name of the container image that was built (as a string)
      
    """
    if isinstance(packages, str):
        packages = [packages]
    elif validate_dict(packages):
        packages = [packages['name']]
        
    if len(packages) == 0:
        raise ValueError("must specify at least one package to build")    
        
    # by default these have an empty string
    if len(skip_tests) == 1 and len(skip_tests[0]) == 0:
        skip_tests = []
        
    if len(test_only) == 1 and len(test_only[0]) == 0:
        test_only = []

    # get default base container (l4t-jetpack)
    if not base:
        base = get_l4t_base()
        
    # add all dependencies to the build tree
    packages = resolve_dependencies(packages)
    print('-- Building containers ', packages)
    
    # make sure all packages can be found before building any
    for package in packages:    
        find_package(package)
            
    # assign default container repository if needed
    if len(name) == 0:   
        name = packages[-1]
    elif name.find(':') < 0 and name[-1] == '/':  # they gave a namespace to build under
        name += packages[-1]
    
    # add prefix to tag
    last_pkg = find_package(packages[-1])
    prefix = last_pkg.get('prefix', '')
    postfix = last_pkg.get('postfix', '')
    
    tag_idx = name.find(':')
    
    if prefix:
        if tag_idx >= 0:
            name = name[:tag_idx+1] + prefix + '-' + name[tag_idx+1:]
        else:
            name = name + ':' + prefix

    if postfix:
        name += f"{':' if tag_idx < 0 else '-'}{postfix}"

    # build chain of all packages
    for idx, package in enumerate(packages):
        # tag this build stage with the sub-package
        container_name = f"{name}-{package.replace(':','_')}"

        # generate the logging file (without the extension)
        log_file = os.path.join(log_dir('build'), container_name.replace('/','_')).replace(':','_')
        
        # build next intermediate container
        pkg = find_package(package)
        
        if 'dockerfile' in pkg:
            cmd = f"{sudo_prefix()}DOCKER_BUILDKIT=0 docker build --network=host --tag {container_name}" + _NEWLINE_
            if no_github_api:
                dockerfilepath = os.path.join(pkg['path'], pkg['dockerfile'])
                with open(dockerfilepath, 'r') as fp:
                    data = fp.read()
                    if 'ADD https://api.github.com' in data:
                        dockerfilepath_minus_github_api = os.path.join(pkg['path'], pkg['dockerfile'] + '.minus-github-api')
                        os.system(f"cp {dockerfilepath} {dockerfilepath_minus_github_api}")
                        os.system(f"sed 's|^ADD https://api.github.com|#[minus-github-api]ADD https://api.github.com|' -i {dockerfilepath_minus_github_api}")
                        cmd += f"--file {os.path.join(pkg['path'], pkg['dockerfile'] + '.minus-github-api')}" + _NEWLINE_
                    else:
                        cmd += f"--file {os.path.join(pkg['path'], pkg['dockerfile'])}" + _NEWLINE_
            else:
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
        if package not in skip_tests and 'intermediate' not in skip_tests and 'all' not in skip_tests:
            if len(test_only) == 0 or package in test_only:
                test_container(container_name, pkg, simulate)
        
        # use this container as the next base
        base = container_name

    # tag the final container
    tag_container(container_name, name, simulate)
    
    # re-run tests on final container
    for package in packages:
        if package not in skip_tests and 'all' not in skip_tests:
            if len(test_only) == 0 or package in test_only:
                test_container(name, package, simulate)
            
    # push container
    if push:
        push_container(name, push, simulate)
            
    return name
    
    
def build_containers(name, packages, base=get_l4t_base(), build_flags='', simulate=False, skip_errors=False, skip_packages=[], skip_tests=[], test_only=[], push=''):
    """
    Build separate container images for each of the requested packages (this is typically used in batch building jobs)
    For example, `['pytorch', 'tensorflow']` would build a pytorch container and a tensorflow container.
    
    TODO add multiprocessing parallel build support for jobs=-1 (use all CPU cores)
    
    Parameters:
      name (str) -- name of container to build (or a namespace to build under, ending in /)
                    if empty, a default name will be assigned based on the package(s) selected.
                    wildcards can be used to select packages (i.e. 'ros*' would build all ROS packages)             
      packages (list[str]) -- list of package names to build (in separated containers)
      base (str) -- base container image to use (defaults to l4t-base or l4t-jetpack)
      build_flags (str) -- arguments to add to the 'docker build' command
      simulate (bool) -- if true, just print out the commands that would have been run
      skip_errors (bool) -- proceed with building the next container on an error (default false)
      skip_packages (list[str]) -- list of packages to skip from the list
      skip_tests (list[str]) -- list of tests to skip (or 'all' or 'intermediate')
      test_only (list[str]) -- only test these specified packages, skipping all other tests
      push (str) -- name of repository or user to push container to (no push if blank)
      
    Returns: 
      True if all containers built successfully, or False if there were any errors.
    """
    if not packages:  # build everything (for testing)
        packages = sorted(find_packages([]).keys())
    
    packages = find_packages(packages, skip=skip_packages)
    print('-- Building containers', list(packages.keys()))
    
    status = {}

    for package in packages:
        try:
            container_name = build_container(name, package, base, build_flags, simulate, skip_tests, test_only, push) 
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
    Tag a container image (source -> target)
    """
    cmd = f"{sudo_prefix()}docker tag {source} {target}"
    
    print(f"-- Tagging container {source} -> {target}")
    print(f"{cmd}\n")
    
    if not simulate:
        subprocess.run(cmd, shell=True, check=True)
        

def push_container(name, repository='', simulate=False):
    """
    Push container to a repository or user with 'docker push'  

    If repository is specified (for example, a DockerHub username) the container will be re-tagged
    under that repository first. Otherwise, it's assumed the image is tagged under the correct name already.
    
    It's also assumed that this machine has already been logged into the repository with 'docker login'
    
    Returns the container name/tag that was pushed.
    """
    cmd = ""
    
    if repository:
        namespace_idx = name.find('/')
        local_name = name
        
        if namespace_idx >= 0:
            name = repository + local_name[namespace_idx:]
        else:
            name = repository + '/' + local_name
        
        print(f"-- Tagging container {local_name} -> {name}")
        
        cmd += f"{sudo_prefix()}docker rmi {name} ; "
        cmd += f"{sudo_prefix()}docker tag {local_name} {name} && "
        
    cmd += f"{sudo_prefix()}docker push {name}"
    
    print(f"-- Pushing container {name}")
    print(f"\n{cmd}\n")
    
    if not simulate:
        subprocess.run(cmd, executable='/bin/bash', shell=True, check=True)
        print(f"\n-- Pushed container {name}\n")
        
    return name
    
    
def test_container(name, package, simulate=False):
    """
    Run tests on a container
    """
    package = find_package(package)
    
    if 'test' not in package:
        return True
        
    for test in package['test']:
        test_cmd = test.split(' ')  # test could be a command with arguments
        test_exe = test_cmd[0]      # just get just the script/executable name
        test_ext = os.path.splitext(test_exe)[1]
        log_file = os.path.join(log_dir('test'), f"{name.replace('/','_')}_{test_exe}").replace(':','_')

        cmd = f"{sudo_prefix()}docker run -t --rm --runtime=nvidia --network=host" + _NEWLINE_
        cmd += f"--volume {package['path']}:/test" + _NEWLINE_
        cmd += f"--volume {os.path.join(_PACKAGE_ROOT, 'data')}:/data" + _NEWLINE_
        cmd += f"--workdir /test" + _NEWLINE_
        cmd += name + _NEWLINE_
        
        cmd += "/bin/bash -c '"
        
        if test_ext == ".py":
            cmd += f"python3 {test}"
        elif test_ext == ".sh":
            cmd += f"/bin/bash {test}"
        else:
            cmd += f"{test}"
        
        cmd += "'" + _NEWLINE_
        cmd += f"2>&1 | tee {log_file + '.txt'}" + "; exit ${PIPESTATUS[0]}"
                
        print(f"-- Testing container {name} ({package['name']}/{test})")
        print(f"\n{cmd}\n")
        
        with open(log_file + '.sh', 'w') as cmd_file:
            cmd_file.write('#!/usr/bin/env bash\n\n')
            cmd_file.write(cmd + '\n')
            
        if not simulate:  # TODO: return false on errors 
            status = subprocess.run(cmd.replace(_NEWLINE_, ' '), executable='/bin/bash', shell=True, check=True)
            
    return True
    
    
_LOCAL_CACHE=[]
_REGISTRY_CACHE=[]


def get_local_containers():
    """
    Get the locally-available container images from the 'docker images' command
    Returns a list of dicts with entries like the following:
    
        {"Containers":"N/A","CreatedAt":"2023-07-23 15:24:28 -0400 EDT","CreatedSince":"42 hours ago",
         "Digest":"\u003cnone\u003e","ID":"6acd9e526f50","Repository":"runner/l4t-pytorch",
         "SharedSize":"N/A","Size":"11.4GB","Tag":"r35.2.1","UniqueSize":"N/A","VirtualSize":"11.37GB"}   
         
    These containers are sorted by most recent created to the oldest.
    """
    global _LOCAL_CACHE
    
    if len(_LOCAL_CACHE) > 0:
        return _LOCAL_CACHE
        
    cmd = ["docker", "images", "--format", "'{{json . }}'"]
    
    if needs_sudo():
        cmd = ["sudo"] + cmd
        
    status = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            #capture_output=True, universal_newlines=True, 
                            shell=False, check=True)

    _LOCAL_CACHE = [json.loads(txt.lstrip("'").rstrip("'"))
        for txt in status.stdout.decode('ascii').splitlines()]
            
    return _LOCAL_CACHE
        

def get_registry_containers(user='dustynv', **kwargs):
    """
    Fetch a DockerHub user's public container images/tags.
    Returns a list of dicts with keys like 'namespace', 'name', and 'tags'.

    To view the number of requests remaining within the rate-limit:
      curl -i https://hub.docker.com/v2/namespaces/dustynv/repositories/l4t-pytorch/tags
      
    All the caching is to prevent going over the DockerHub API rate limits.
    """
    global _REGISTRY_CACHE
    
    if len(_REGISTRY_CACHE) > 0:
        return _REGISTRY_CACHE
    
    cache_path = kwargs.get('registry_cache',
        os.environ.get('DOCKERHUB_CACHE', 
            os.path.join(_PACKAGE_ROOT, 'data/containers.json')
    ))
    
    cache_enabled = (cache_path != "0" and cache_path.lower() != "off")

    if cache_enabled and os.path.isfile(cache_path):
        if time.time() - os.path.getmtime(cache_path) > 600 and os.geteuid() != 0:
            cmd = f"cd {_PACKAGE_ROOT} && git fetch origin dev --quiet && git checkout --quiet origin/dev -- {os.path.relpath(cache_path, _PACKAGE_ROOT)}"
            status = subprocess.run(cmd, executable='/bin/bash', shell=True, check=False)
            if status.returncode != 0:
                logging.error(f'failed to update container registry cache from GitHub ({cache_path})')
                logging.error(f'return code {status.returncode} > {cmd}')
                
        with open(cache_path) as cache_file:
            try:
                _REGISTRY_CACHE = json.load(cache_file)
                return _REGISTRY_CACHE
            except Exception:
                pass
                
    hub = dockerhub_api.DockerHub(return_lists=True, token=os.environ.get('DOCKERHUB_TOKEN'))
    _REGISTRY_CACHE = hub.repositories(user)
    
    for repo in _REGISTRY_CACHE:
        repo['tags'] = hub.tags(user, repo['name'])

    if not cache_enabled:
        cache_path = 'data/containers.json'
        
    with open(cache_path, 'w') as cache_file:
        json.dump(_REGISTRY_CACHE, cache_file, indent=2)

    return _REGISTRY_CACHE
    
  
def find_local_containers(package, return_dicts=False, **kwargs):
    """
    Search for local containers on the machine containing this package.
    Returns a list of strings, unless return_dicts=True in which case
    a list of dicts is returned with the full metadata from the Docker engine.
    """
    if isinstance(package, dict):
        package = package['name']
    
    namespace, repo, tag = split_container_name(package)
    local_images = get_local_containers()
    pprint_debug(local_images)
    
    found_containers = []
    
    for image in local_images:
        if namespace:
            if image['Repository'] != f'{namespace}/{repo}':
                continue
        else:
            if image['Repository'].split('/')[-1] != repo:
                continue

        if tag and tag != image['Tag'] and not image['Tag'].startswith(tag + '-'):
            continue
            
        if return_dicts:
            found_containers.append(image)
        else:
            found_containers.append(f"{image['Repository']}:{image['Tag']}")
            
    return found_containers
    
        
def find_registry_containers(package, check_l4t_version=True, return_dicts=False, **kwargs):
    """
    Search DockerHub for container images compatible with the package or container name.
    
    The returned list of images will also be compatible with the version of L4T 
    running on the device, unless check_l4t_version is set to false
    
    Normally a list of strings is returned, unless return_dicts=True in which case
    a list of dicts is returned with the full metadata from DockerHub.
    """
    if isinstance(package, dict):
        package = package['name']
    
    namespace, repo, tag = split_container_name(package)
    registry_repos = get_registry_containers(**kwargs)
    pprint_debug(registry_repos)

    found_containers = []
    
    for registry_repo in registry_repos:
        if registry_repo['name'] != repo:
            continue
        
        repo_copy = copy.deepcopy(registry_repo)
        repo_copy['tags'] = []
        
        for registry_image in registry_repo['tags']:
            if tag and not (tag == registry_image['name'] or fnmatch.fnmatch(registry_image['name'], tag + '-*')):
                continue
            
            if check_l4t_version:
                if not l4t_version_compatible(l4t_version_from_tag(registry_image['name']), **kwargs):
                    continue
            
            repo_copy['tags'].append(copy.deepcopy(registry_image))
            
            if not return_dicts:
                found_containers.append(
                    f"{registry_repo['namespace']}/{registry_repo['name']}:{registry_image['name']}"
                )
        
        if return_dicts and len(repo_copy['tags']) > 0:
            found_containers.append(repo_copy)
            
    return found_containers
    
            
def find_container(package, prefer_sources=['local', 'registry', 'build'], disable_sources=[], quiet=True, **kwargs):
    """
    Finds a local or remote container image to run for the given package (returns a string)
    TODO search for other packages that depend on this package if an image isn't available.
    TODO check if the dockerhub image has updated vs local copy, and if so ask user if they want to pull it.
    """
    if isinstance(package, dict):
        package = package['name']
     
    namespace, repo, tag = split_container_name(package)
    log_debug(f"-- Finding compatible container image for namespace={namespace} repo={repo} tag={tag}")

    for source in prefer_sources:
        if source in disable_sources:
            continue
            
        if source == 'local':
            local_images = find_local_containers(package, **kwargs)
            
            if len(local_images) > 0:
                return local_images[0]
     
        elif source == 'registry':
            registry_images = find_registry_containers(package, return_dicts=True, **kwargs)
            
            if len(registry_images) > 0:
                img = registry_images[0]  # TODO allow use to select image if there are multiple candidates
                img_tag = img['tags'][0]
                img_name = f"{img['namespace']}/{img['name']}:{img_tag['name']}"
                if quiet or query_yes_no(f"\nFound compatible container {img_name} ({img_tag['tag_last_pushed'][:10]}, {img_tag['full_size']/(1024**3):.1f}GB) - would you like to pull it?", default="yes"):
                    return img_name
        
        elif source == 'build':
            if not quiet and query_yes_no(f"\nCouldn't find a compatible container for {package}, would you like to build it?"):
                return build_container('', package) #, simulate=True)

    # compatible container image could not be found
    return None
