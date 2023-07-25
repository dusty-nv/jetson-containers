#!/usr/bin/env python3
import sys


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
