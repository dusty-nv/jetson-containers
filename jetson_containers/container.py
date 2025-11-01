#!/usr/bin/env python3
import copy
import datetime
import dockerhub_api
import fnmatch
import json
import os
import pprint
import shutil
import subprocess
import sys
import time
import traceback
from packaging.version import Version
from typing import List, Dict, Any, Union

import logging
from .l4t_version import (
    L4T_VERSION, LSB_RELEASES, IS_TEGRA, IS_SBSA, l4t_version_from_tag, l4t_version_compatible,
    get_l4t_base, get_cuda_arch, get_cuda_version, get_jetpack_version, get_lsb_release
)
from .logging import (
    get_log_dir, log_status, log_success, log_status, log_warning, log_debug,
    log_block, log_info, print_log, pprint_debug, colorize, LogConfig
)
from .packages import find_package, find_packages, resolve_dependencies, validate_dict
from .utils import (
    split_container_name, query_yes_no, needs_sudo, sudo_prefix,
    get_env, get_dir, get_repo_dir
)

_NEWLINE_=" \\\n"  # used when building command strings

def format_time(seconds):
    """Format time in hh:mm:ss format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_time_minutes(seconds):
    """Format time in mm:ss format, without padding for minutes over 99"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    if minutes < 100:
        return f"{minutes:02d}m{seconds:02d}s"
    else:
        return f"{minutes}m{seconds:02d}s"

class BuildTimer:
    def __init__(self):
        self.start_time = time.time()
        self.stage_start = time.time()
        self.current_stage = 0

    def get_elapsed(self):
        """Get total elapsed time"""
        return time.time() - self.start_time

    def next_stage(self):
        """Move to next stage and reset stage timer"""
        self.stage_start = time.time()
        self.current_stage += 1

def build_container(
        name: str='', packages: list=[], base: str=get_l4t_base(),
        build_flags: str='', build_args: dict=None, simulate: bool=False,
        skip_packages: list=[], skip_tests: list=[], test_only: list=[],
        push: str='', no_github_api=False, **kwargs
    ):
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
      skip_packages (list[str]) -- list of packages to skip from the build
      skip_tests (list[str]) -- list of tests to skip (or 'all' or 'intermediate')
      test_only (list[str]) -- only test these specified packages, skipping all other tests
      push (str) -- name of repository or user to push container to (no push if blank)
      no_github_api (bool) -- if true, use custom Dockerfile with no `ADD https://api.github.com/repos/...` line.

    Returns:
      The full name of the container image that was built (as a string)

    """

    # Start timing at the very beginning
    build_start_time = time.time()

    try:

        if isinstance(packages, str):
            packages = [packages]
        elif validate_dict(packages):
            packages = [packages['name']]
        else:
            packages = packages.copy()

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
        packages = resolve_dependencies(packages, skip_packages=skip_packages)

        # make sure all packages can be found before building any
        for package in packages:
            find_package(package)

        # assign default container repository if needed
        if len(name) == 0:
            name = packages[-1]
            repo_name = packages[-1]
        elif name.find(':') < 0 and name[-1] == '/':  # they gave a namespace to build under
            name += packages[-1]
            repo_name = packages[-1]
        else:
            repo_name = name.split(':')[0].split('/')[-1]

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

        # Sanitize name for Docker (replace + with - since Docker doesn't allow + in tags)
        name = name.replace('+', '-')

        log_status(f'<b>BUILDING  {packages}</b>')

        # Add N-second countdown with BUILD_DELAY=N environment variable
        build_delay = get_env('BUILD_DELAY', default=0, type=int)

        if build_delay > 0:
            log_info("Starting build in...")
            for i in range(build_delay, 0, -1):
                log_info(f"{i}...")
                time.sleep(1)

        # Initialize status bar and clear screen
        terminal = shutil.get_terminal_size(fallback=(80, 24))
        print(f'\033[1;{terminal.lines-1}r\033[?6l\033[2J\033[H]', end='', flush=True)
        LogConfig.status = True

        # Initialize build timer
        timer = BuildTimer()

        # build chain of all packages
        for idx, package in enumerate(packages):
            pkg = find_package(package)
            # tag this build stage with the sub-package
            container_name = f"{name}-{package.replace(':','_')}"

            # generate the logging file (without the extension)
            log_file = os.path.join(get_log_dir('build'), f"{idx+1:02d}o{len(packages)}_{container_name.replace('/','_')}").replace(':','_')
            jetpack_version = get_jetpack_version()
            if 'dockerfile' in pkg:
                # Check if SSH key is provided for SCP uploads - BuildKit/buildx is required for secret mounting
                scp_upload_key = os.environ.get('SCP_UPLOAD_KEY')
                scp_key_provided = scp_upload_key and os.path.isfile(scp_upload_key)

                # Use buildx which always uses BuildKit and has better log size limit support
                # Enable buildx if SSH key is provided (required for secrets) or user prefers it
                user_buildkit = os.environ.get('DOCKER_BUILDKIT', '0')
                use_buildx = user_buildkit != '0' or scp_key_provided

                # Set BuildKit log size limit (default 500MB, configurable via BUILDKIT_STEP_LOG_MAX_SIZE)
                # Default Docker BuildKit limit is 2MiB which can clip large build outputs
                buildkit_log_size = os.environ.get('BUILDKIT_STEP_LOG_MAX_SIZE', '524288000')  # 500MB default

                if use_buildx:
                    # Use buildx with log size configuration
                    buildkit_env = f"BUILDKIT_STEP_LOG_MAX_SIZE={buildkit_log_size}"
                    cmd = f"{sudo_prefix()}{buildkit_env} docker buildx build --network=host --shm-size=8g" + _NEWLINE_
                    cmd += f"  --progress=plain" + _NEWLINE_
                    cmd += f"  --load" + _NEWLINE_  # Load image into local Docker daemon
                else:
                    # Fallback to regular docker build
                    buildkit_env = f"DOCKER_BUILDKIT=0"
                    cmd = f"{sudo_prefix()}{buildkit_env} docker build --network=host --shm-size=8g" + _NEWLINE_
                cmd += f"  --tag {container_name}" + _NEWLINE_
                if no_github_api:
                    dockerfilepath = os.path.join(pkg['path'], pkg['dockerfile'])
                    with open(dockerfilepath, 'r') as fp:
                        data = fp.read()
                        if 'ADD https://api.github.com' in data:
                            dockerfilepath_minus_github_api = os.path.join(pkg['path'], pkg['dockerfile'] + '.minus-github-api')
                            os.system(f"cp {dockerfilepath} {dockerfilepath_minus_github_api}")
                            os.system(f"sed 's|^ADD https://api.github.com|#[minus-github-api]ADD https://api.github.com|' -i {dockerfilepath_minus_github_api}")
                            cmd += f"  --file {os.path.join(pkg['path'], pkg['dockerfile'] + '.minus-github-api')}" + _NEWLINE_
                        else:
                            cmd += f"  --file {os.path.join(pkg['path'], pkg['dockerfile'])}" + _NEWLINE_
                else:
                    cmd += f"  --file {os.path.join(pkg['path'], pkg['dockerfile'])}" + _NEWLINE_

                cmd += f"  --build-arg BASE_IMAGE={base}" + _NEWLINE_
                cmd += f"  --build-arg NVIDIA_DRIVER_CAPABILITIES=all" + _NEWLINE_

                if 'build_args' in pkg:
                    cmd += ''.join([f"  --build-arg {key}=\"{value}\"" + _NEWLINE_ for key, value in pkg['build_args'].items()])

                if build_args:
                    for key, value in build_args.items():
                        cmd += f"  --build-arg {key}={value}" + _NEWLINE_

                if 'build_flags' in pkg:
                    cmd += '  ' + pkg['build_flags'] + _NEWLINE_

                if build_flags:
                    cmd += '  ' + build_flags + _NEWLINE_

                # Add SSH key secret mount if provided (requires BuildKit/buildx)
                if scp_key_provided:
                    if not use_buildx:
                        log_warning("SCP_UPLOAD_KEY requires BuildKit/buildx - using buildx")
                        use_buildx = True
                    cmd += f"  --secret id=scp_upload_key,src={scp_upload_key}" + _NEWLINE_
                    # Update SCP_UPLOAD_KEY build arg to point to permanent location after secret is copied
                    # Check if it's not already set in package build_args or user build_args
                    scp_key_in_pkg = 'build_args' in pkg and 'SCP_UPLOAD_KEY' in pkg['build_args']
                    scp_key_in_build = build_args and 'SCP_UPLOAD_KEY' in build_args
                    if not scp_key_in_pkg and not scp_key_in_build:
                        # Set to /root/.ssh/scp_upload_key which is where Dockerfile.pip will copy it
                        cmd += "  --build-arg SCP_UPLOAD_KEY=/root/.ssh/scp_upload_key" + _NEWLINE_

                cmd += '   ' + pkg['path']

                log_block(f"<b>> BUILDING  {container_name}</b>", f"<b>{cmd}</b>")

                # Calculate spaces needed to align time to right edge
                status_text = f"[{idx+1}/{len(packages)}] Building {package} ({container_name})"
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                time_text = f"{idx} stages completed in {format_time_minutes(timer.get_elapsed())} at {current_time}"
                spaces_needed = terminal.columns - len(status_text) - len(time_text)
                if spaces_needed > 0:
                    status_text = status_text + ' ' * spaces_needed
                log_status(f"{status_text}{time_text}")

                cmd += _NEWLINE_ + f"2>&1 | tee {log_file + '.txt'}" + "; exit ${PIPESTATUS[0]}"  # non-tee version:  https://stackoverflow.com/a/34604684

                with open(log_file + '.sh', 'w') as cmd_file:   # save the build command to a shell script for future reference
                    cmd_file.write('#!/usr/bin/env bash\n\n')
                    cmd_file.write(cmd + '\n')

                if not simulate:  # remove the line breaks that were added for readability, and set the shell to bash so we can use $PIPESTATUS
                    status = subprocess.run(cmd.replace(_NEWLINE_, ' '), executable='/bin/bash', shell=True, check=True)
                    print('')
            else:
                tag_container(base, container_name, simulate)

            # run tests on the intermediate container
            if package not in skip_tests and 'intermediate' not in skip_tests and 'all' not in skip_tests:
                if len(test_only) == 0 or package in test_only:
                    status_text = f"[{idx+1}/{len(packages)}] Testing {package} ({container_name})"
                    current_time = datetime.datetime.now().strftime("%H:%M:%S")
                    time_text = f"{idx} stages completed in {format_time_minutes(timer.get_elapsed())} at {current_time}"
                    spaces_needed = terminal.columns - len(status_text) - len(time_text)
                    if spaces_needed > 0:
                        status_text = status_text + ' ' * spaces_needed
                    log_status(f"{status_text}{time_text}")
                    test_container(container_name, pkg, simulate, build_idx=idx)

            # use this container as the next base
            base = container_name
            timer.next_stage()

        # tag the final container
        tag_container(container_name, name, simulate)

        # re-run tests on final container
        for idx, package in enumerate(packages):
            if package not in skip_tests and 'all' not in skip_tests:
                if len(test_only) == 0 or package in test_only:
                    status_text = f"[{idx+1}/{len(packages)}] Testing {package} ({name})"
                    current_time = datetime.datetime.now().strftime("%H:%M:%S")
                    time_text = f"{idx} stages completed in {format_time_minutes(timer.get_elapsed())} at {current_time}"
                    spaces_needed = terminal.columns - len(status_text) - len(time_text)
                    if spaces_needed > 0:
                        status_text = status_text + ' ' * spaces_needed
                    log_status(f"{status_text}{time_text}")
                    test_container(name, package, simulate, build_idx=idx)
                    timer.next_stage()

        # push container
        if push:
            log_status(f'Pushing {name}')
            name = push_container(name, push, simulate)

        # Calculate total build time
        build_end_time = time.time()
        total_duration = build_end_time - build_start_time

        log_success('=====================================================================================')
        log_success('=====================================================================================')
        log_success(f'✅ <b>`jetson-containers build {repo_name}`</b> ({name})')
        log_success(f'⏱️  Total build time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)')
        log_success('=====================================================================================')
        log_success('=====================================================================================')

        return name

    except Exception as e:
        # Calculate time even if build failed
        build_end_time = time.time()
        total_duration = build_end_time - build_start_time

        log_warning('=====================================================================================')
        log_warning('=====================================================================================')
        log_warning(f'💣 `jetson-containers build` failed after {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)')
        log_warning(f'Error: {str(e)}')
        log_warning('=====================================================================================')
        log_warning('=====================================================================================')

        # Re-raise the exception so the calling code knows it failed
        raise

def build_containers(
        name: str='',
        packages: list=[],
        skip_packages: list=[],
        skip_errors: bool=False,
        **kwargs
    ):
    """
    Build separate container images for each of the requested packages (this is typically used in batch building jobs)
    For example, `['pytorch', 'tensorflow']` would build a pytorch container and a tensorflow container.

    Parameters:
      name (str) -- name of container to build (or a namespace to build under, ending in /)
                    if empty, a default name will be assigned based on the package(s) selected.
                    wildcards can be used to select packages (i.e. 'ros*' would build all ROS packages)
      packages (list[str]) -- list of package names to build (in separated containers)
      skip_packages (list[str]) -- list of packages to skip from the list
      skip_errors (bool) -- proceed with building the next container on an error (default false)

    kwargs:
      See the keyword arguments that are passed through to the build_containers() function.

    Returns:
      True if all containers built successfully, or False if there were any errors.
    """
    status = {} # pass/fail result of each build

    if not packages:  # build everything (for testing)
        packages = sorted(find_packages([]).keys())

    packages = find_packages(packages, skip=skip_packages)

    for package in packages:
        try:
            container_name = build_container(name, package, **kwargs)
        except Exception as error:
            container_name = name
            print(error)
            if not skip_errors:
                return False #raise error #sys.exit(os.EX_SOFTWARE)
            status[package] = (container_name, error)
        else:
            status[package] = (container_name, None)

    log_info(f"Build logs at:  {get_log_dir('build')}")

    for package, (container_name, error) in status.items():
        msg = f"   * {package} ({container_name}) {'FAILED' if error else 'SUCCESS'}"
        if error is not None:
            msg += f"  ({error})"
        print_log(msg, level='error' if error else 'success')

    for _, error in status.values():
        if error:
            return False

    return True


def tag_container(source, target, simulate=False):
    """
    Tag a container image (source -> target)
    """
    cmd = f"{sudo_prefix()}docker tag {source} \\\n           {target}"

    log_block(f"<b>Tagging {source} </b>\n<b>     as {target}</b>\n", f"<b>{cmd}</b>\n")

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

        cmd += f"{sudo_prefix()}docker rmi {name} ; "
        cmd += f"{sudo_prefix()}docker tag {local_name} {name} && "

    cmd += f"{sudo_prefix()}docker push {name}"

    log_block(f"<b>PUSHING {name}</b>", f"<b>{cmd}</b>\n")
    log_status(f"Pushing {name}")

    if not simulate:
        subprocess.run(cmd, executable='/bin/bash', shell=True, check=True)
        log_success(f"Pushed container {name}\n")

    return name


def test_container(name, package, simulate=False, build_idx=None):
    """
    Run tests on a container
    """
    package = find_package(package)

    if 'test' not in package:
        return True

    for idx, test in enumerate(package['test']):
        test_cmd = test.split(' ')  # test could be a command with arguments
        test_exe = test_cmd[0]      # just get just the script/executable name
        test_ext = os.path.splitext(test_exe)[1]
        log_file = os.path.join(get_log_dir('test'), f"{build_idx+1:02d}-{idx+1}_{name.replace('/','_')}_{test_exe}").replace(':','_')

        cmd = f"{sudo_prefix()}docker run -t --rm --network=host --privileged --shm-size=8g "

        cmd += f"--gpus=all" + _NEWLINE_
        cmd += f"  --env NVIDIA_DRIVER_CAPABILITIES=all" + _NEWLINE_

        cmd += f"  --volume {package['path']}:/test" + _NEWLINE_
        cmd += f"  --volume {get_dir('data')}:/data" + _NEWLINE_
        cmd += '  ' + name + _NEWLINE_

        cmd += "    /bin/bash -c '"

        if test_ext == ".py":
            cmd += f"python3 /test/{test}"
        elif test_ext == ".sh":
            cmd += f"/bin/bash /test/{test}"
        else:
            cmd += f"/test/{test}"

        log_block(f"<b>> TESTING  {name}</b>", f"<b>{cmd}</b>\n")

        cmd += "'" + _NEWLINE_
        cmd += f"2>&1 | tee {log_file + '.txt'}" + "; exit ${PIPESTATUS[0]}"

        with open(log_file + '.sh', 'w') as cmd_file:
            cmd_file.write('#!/usr/bin/env bash\n\n')
            cmd_file.write(cmd + '\n')

        if not simulate:  # TODO: return false on errors
            status = subprocess.run(cmd.replace(_NEWLINE_, ' '), executable='/bin/bash', shell=True, check=True)
            print('')

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

    pprint_debug(_LOCAL_CACHE)

    return _LOCAL_CACHE


def get_registry_containers(user='dustynv', use_cache=True, **kwargs):
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
            os.path.join(get_dir('data'), 'containers.json')
    ))

    has_cache_path = (cache_path != "0" and cache_path.lower() != "off")
    cache_enabled = (use_cache and has_cache_path)

    if cache_enabled and os.path.isfile(cache_path):
        if time.time() - os.path.getmtime(cache_path) > 600 and os.geteuid() != 0:
            cmd = f"cd {get_repo_dir()} && git fetch origin dev --quiet && git checkout --quiet origin/dev -- {os.path.relpath(cache_path, get_repo_dir())}"
            status = subprocess.run(cmd, executable='/bin/bash', shell=True, check=False)
            if status.returncode != 0:
                logging.error(f'failed to update container registry cache from GitHub ({cache_path})')
                logging.error(f'return code {status.returncode} > {cmd}')

        with open(cache_path) as cache_file:
            try:
                _REGISTRY_CACHE = json.load(cache_file)
                pprint_debug(_REGISTRY_CACHE)
                return _REGISTRY_CACHE
            except Exception:
                pass

    hub = dockerhub_api.DockerHub(return_lists=True, token=os.environ.get('DOCKERHUB_TOKEN'))
    _REGISTRY_CACHE = hub.repositories(user)

    for repo in _REGISTRY_CACHE:
        repo['tags'] = hub.tags(user, repo['name'])

    if not has_cache_path:
        cache_path = 'data/containers.json'

    with open(cache_path, 'w') as cache_file:
        json.dump(_REGISTRY_CACHE, cache_file, indent=2)

    pprint_debug(_REGISTRY_CACHE)
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
    log_debug(f"Finding compatible container image for namespace={namespace} repo={repo} tag={tag}")

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

def parse_container_versions(tags, use_defaults=True, **kwargs):
    """
    Parse well-formed container tags into their L4T_VERSION, CUDA_VERSION, LSB_RELEASE, ect.
    This returns a dict of the aformentioned versions (typically from l4t_version.py)
    Missing tags will be filled in with their defaults unless ``use_defaults=False``
    """
    #from ..packages.ros.version import ROS_PACKAGES
    ROS_PACKAGES = ['ros_base', 'ros_core', 'desktop']  # TODO add function to import package

    container = tags.lower()

    if ':' in tags:
        tags = tags.split(':')[-1]

    tags = tags.split('-')
    data = {}

    for x in tags:
        if not x or len(x) == 0:
            continue
        if len(x) >= 4 and x.startswith('cu') and x[2:].isnumeric():
            data['CUDA_VERSION'] = f"{float(x[2:])/10:.1f}"
        elif len(x) >= 3 and x.lower().startswith('r') and x[1:3].isnumeric():
            data['L4T_VERSION'] = x[1:]
        elif len(x) == 5 and x in LSB_RELEASES:
            data['LSB_RELEASE'] = x
        elif 'ros' in container and x in ROS_PACKAGES:
            data['ROS_PACKAGE'] = x
        elif 'version' not in data:
            data['version'] = x
        else:
            log_info(f"Skipping unknown container tag '{x}' while parsing '{container}'")

    if not use_defaults:
        return data

    if not 'L4T_VERSION' in data:
        log_warning(f"Missing L4T_VERSION tag from container '{container}'")
        return data

    l4t_version = data['L4T_VERSION']

    data.setdefault('JETPACK_VERSION', get_jetpack_version(l4t_version=l4t_version))
    data.setdefault('CUDA_VERSION', get_cuda_version(l4t_version=l4t_version))
    data.setdefault('CUDA_ARCH', get_cuda_arch(l4t_version=l4t_version, format=str))
    data.setdefault('LSB_RELEASE', get_lsb_release(l4t_version=l4t_version))

    if 'ros' in container and 'ROS_PACKAGE' not in data:
        for ros_package in ROS_PACKAGES:
            if ros_package in container or ros_package.replace('_', '-') in container:
                data['ROS_PACKAGE'] = ros_package
                break

    for k,v in data.items():
        if not v:
            del k
            continue
        data[k] = str(v)

    return data
