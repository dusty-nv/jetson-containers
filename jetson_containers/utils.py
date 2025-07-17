#!/usr/bin/env python3
import functools
import grp
import os
import pprint
import requests
import subprocess
import sys
import time
from typing import Dict, Literal


def check_dependencies(install: bool = True):
    """
    Check if the required pip packages are available, and install them if needed.
    """
    try:
        import yaml
        import wget
        import dockerhub_api
        import tabulate
        import termcolor

        from packaging.version import Version

        x = Version('1.2.3') # check that .major, .minor, .micro are available
        x = x.major          # (these are in packaging>=20.0)

    except Exception as error:
        if not install:
            raise error

        requirements = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'requirements.txt')
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', requirements]

        print('-- Installing required packages:', cmd)
        subprocess.run(cmd, shell=False, check=True)


def get_dir(key, root=None):
    """
    Find the absolute paths to common project directories.
    """
    if not root:
        root = get_repo_dir()

    key = key.lower()

    if key == 'repo':
        return root

    return os.path.join(root, key)


def get_repo_dir():
    """
    Returns the path of the jetson-containers git repo
    (this is typically one directory up from this file)
    """
    return os.path.dirname(os.path.dirname(__file__))


def get_env(key, default=None, type=str):
    """
    Retrieve an environment variable and convert it to the given type,
    or return the default value if it's undefined (by default `None`)

    If key is a list or tuple, then the first present key will be used.
    For example, this will return `$CACHE_DIR` first and so forth:

      get_env(('CACHE_DIR', 'HF_HOME', 'XGD_HOME'), '/root/.cache')

    If none of those keys are found, then the default value is returned.
    """
    if not key:
        return default

    if isinstance(key, str):
        key = [key]

    def find_env(keys):
        for k in keys:
            if k not in os.environ:
                continue
            v = os.environ[k]
            if v is None or len(v) == 0:
                continue
            return v

    value = find_env(key)

    if value is None:
        return default
    elif type is None or type == str:
        return value
    elif type == bool:
        return to_bool(value)
    else:
        try:
            return type(value)
        except Exception as error:
            print(f"-- Warning:  exception occurred parsing environment variable `${key} = {value}` (default={default}, type={type})\n--           {error}")
            return default


def get_env_flag(key, default: bool=False) -> bool:
    """
    Return a boolean environment variable parsed from a truthy string
    or value as converted by the `to_bool()` function, for example:

       * DEBUG=ON  get_env_flag('DEBUG') => True
       * DEBUG=0   get_env_flag('DEBUG') => False

    This can also handle evaluating multiple keys as a list/tuple:

       * VERBOSE=1  get_env_flag(('DEBUG', 'VERBOSE')) => True
       * A=ON B=no  get_env_flag(['A','B'], False) => True

    In that case, the values of multiple flags are OR'd together (|)
    """
    return get_env(key, default=default, type=bool)


def to_bool(value: str, default: None) -> bool:
    """
    Convert a truthy value or string (like 1, 'true', 'ON', ect) to boolean.
    """
    value = str(value).lower()
    falsy = ['off', 'false', '0', 'no', 'disabled', 'none']
    truthy = ['on', 'true', '1', 'yes', 'enabled']

    if value in truthy:
        return True
    elif value in falsy:
        return False
    elif default is not None:
        return default
    else:
        raise ValueError(f"expected a truthy value, got '{value}'")


def query_yes_no(question: str, default: Literal["no", "yes"] = "no") -> bool:
    """
    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def split_container_name(name):
    """
    Splits a container name like `dustynv/ros:tag` or `nvcr.io/nvidia/l4t-pytorch:tag`
    into a (namespace, repository, tag) tuple (where namespace would be `dustynv` or
    `nvcr.io/nvidia`, and repository would be `ros` or `l4t-pytorch`)
    """
    parts = name.split(':')
    repo = parts[0]
    namespace = ''
    tag = ''

    if len(parts) == 2:
        tag = parts[1]

    parts = repo.split('/')

    if len(parts) == 2:
        namespace = parts[0]
        repo = parts[1]

    return namespace, repo, tag


def user_in_group(group) -> bool:
    """
    Returns true if the user running the current process is in the specified user group.
    Equivalent to this bash command:   id -nGz "$USER" | grep -qzxF "$GROUP"
    """
    try:
        group = grp.getgrnam(group)
    except KeyError:
        return False

    return (group.gr_gid in os.getgroups())


def is_root_user() -> bool:
    """
    Returns true if this is the root user running
    """
    return os.geteuid() == 0


def sudo_prefix(group: str = 'docker'):
    """
    Returns a sudo prefix for command strings if the user needs sudo for accessing docker
    """
    return "sudo " if needs_sudo(group) else ""


def needs_sudo(group: str='docker') -> bool:
    """
    Returns true if sudo is needed to use the docker engine (if user isn't in the docker group)
    """
    global NEEDS_SUDO

    if NEEDS_SUDO is not None:
        return NEEDS_SUDO

    def _needs_sudo(group):
        if is_root_user():
            return False
        else:
            if user_in_group(group):
                return False

            try:
                subprocess.run(
                    ['docker', 'info'],
                    check=True, shell=False,
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL
                )
                return False
            except Exception:
                return True

    NEEDS_SUDO = _needs_sudo(group)
    return NEEDS_SUDO


# will be true if user needs sudo to access docker
NEEDS_SUDO=None
