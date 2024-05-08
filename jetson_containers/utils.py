#!/usr/bin/env python3
import os
import grp
import sys
import json
import pprint
import urllib.request
from urllib.request import urlopen, Request


def check_dependencies(install=True):
    """
    Check if the required pip packages are available, and install them if needed.
    """
    try:
        import yaml
        import wget
        import dockerhub_api
        
        from packaging.version import Version
        
        x = Version('1.2.3') # check that .major, .minor, .micro are available
        x = x.major          # (these are in packaging>=20.0)
        
    except Exception as error:
        if not install:
            raise error
            
        import os
        import sys
        import subprocess
        
        requirements = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'requirements.txt')
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', requirements]
        
        print('-- Installing required packages:', cmd)
        subprocess.run(cmd, shell=False, check=True)
        

def query_yes_no(question, default="no"):
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
    
    
def user_in_group(group):
    """
    Returns true if the user running the current process is in the specified user group.
    Equivalent to this bash command:   id -nGz "$USER" | grep -qzxF "$GROUP"
    """
    try:
        group = grp.getgrnam(group)
    except KeyError:
        return False
        
    return (group.gr_gid in os.getgroups())
  

def is_root_user():
    """
    Returns true if this is the root user running
    """
    return os.geteuid() == 0
    
    
def needs_sudo(group='docker'):
    """
    Returns true if sudo is needed to use the docker engine (if user isn't in the docker group)
    """
    if is_root_user():
        return False
    else:
        return not user_in_group(group)
    

def sudo_prefix(group='docker'):
    """
    Returns a sudo prefix for command strings if the user needs sudo for accessing docker
    """
    if needs_sudo(group):
        #print('USER NEEDS SUDO')
        return "sudo "
    else:
        return ""
        
        
def github_latest_commit(repo, branch='main', github_token=None):
    print(f"github_latest_commit({repo}, branch={branch}, github_token={github_token})")
    """
    Returns the SHA of the latest commit to the given github user/repo/branch.
    """
    url = f"https://api.github.com/repos/{repo}/commits/{branch}"
    github_token = os.environ.get('GITHUB_TOKEN')

    if github_token:
        log_debug(f"-- GITHUB_TOKEN={github_token}")
        headers = {'Authorization': 'token %s' % github_token}
        request = Request(url, headers=headers)
    else:
        request = Request(url)
        
    response = urlopen(request)
    data = response.read()
    encoding = response.info().get_content_charset('utf-8')
    msg = json.loads(data.decode(encoding))
    
    return msg['sha']
    
    
def log_debug(*args, **kwargs):
    """
    Debug print function that only prints when VERBOSE or DEBUG environment variable is set
    TODO change this to use python logging APIs or move to logging.py
    """
    if os.environ.get('VERBOSE', False) or os.environ.get('DEBUG', False):
        print(*args, **kwargs)
        
        
def pprint_debug(*args, **kwargs):
    """
    Debug print function that only prints when VERBOSE or DEBUG environment variable is set
    TODO change this to use python logging APIs or move to logging.py
    """
    if os.environ.get('VERBOSE', False) or os.environ.get('DEBUG', False):
        pprint.pprint(*args, **kwargs)
        
