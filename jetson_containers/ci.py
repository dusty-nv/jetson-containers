#!/usr/bin/env python3
#L4T_VERSION=35.2.1 python3 -m jetson_containers.ci generate
import os
import re
import wget
import shutil
import socket
import pprint
import argparse
import subprocess

from jetson_containers import find_packages, L4T_VERSION


def generate_workflow(package, root, l4t_version, simulate=False):
    """
    Generate the YAML workflow definition for building container for that package
    """
    name = package['name']
    workflow_name = f"{name}{'-' if ':' in name else ':'}r{l4t_version}"
    filename = os.path.join(root, '.github/workflows', f"{name.replace(':','_')}-r{l4t_version}.yml")
    
    txt = f"name: \"{workflow_name}\"\n"
    txt += "on:\n"
    txt += "  workflow_dispatch: {}\n"
    txt += "  push:\n"
    txt += "    branches:\n"
    txt += "      - 'dev'\n"
    txt += "jobs:\n"
    txt += f"  {workflow_name.replace(':','_')}:\n"
    txt += f"    runs-on: self-hosted-jetson r{l4t_version}\n"
    txt += "    steps:\n"
    txt += f"    - run: echo \"Building {workflow_name}\"\n"
    
    print(filename)
    print(txt)


def register_runner(token, root, repo, labels=[], prefix='runner', simulate=False):
    """
    Setup and register this machine as a self-hosted runner with GitHub
    """
    if not args.token:
        raise ValueError(f"--token must be provided from GitHub when registering self-hosted runners")
            
    labels.extend([
        'self-hosted-jetson',
        f'r{L4T_VERSION.major}.{L4T_VERSION.minor}',
        f'r{L4T_VERSION}',
        socket.gethostname(),
    ])
    
    labels = [x for x in labels if x]
    
    run_dir = os.path.join(root, prefix)
    run_tar = os.path.join(run_dir, "actions-runner-linux-arm64-2.304.0.tar.gz")
    run_url = "https://github.com/actions/runner/releases/download/v2.304.0/actions-runner-linux-arm64-2.304.0.tar.gz"
    
    if not os.path.isfile(run_tar) and not simulate:
        print(f"-- Installing self-hosted runner under {run_dir}")
        os.makedirs(run_dir, exist_ok=True)
        wget.download(run_url, run_tar)
        shutil.unpack_archive(run_tar, run_dir)
        
    cmd = f"cd {run_dir} && "
    cmd += f"./config.sh --url {repo} --token {token} --labels {','.join(labels)} --unattended && sudo ./svc.sh install && sudo ./svc.sh status "

    print(f"-- Configuring self-hosted runner for {repo}\n")
    print(cmd)
    
    if not simulate:
        subprocess.run(cmd, executable='/bin/bash', shell=True, check=True) 
    
    # https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/configuring-the-self-hosted-runner-application-as-a-service
    print(f"\n-- Commands for interacting with the runner service:")
    print(f"  cd {run_dir}\n")
    print(f"  sudo ./svc.sh start      # manually start the service")
    print(f"  sudo ./svc.sh stop       # manually stop the service")
    print(f"  sudo ./svc.sh status     # check the service status")
    print(f"  sudo ./svc.sh uninstall  # remove systemd service\n")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('cmd', type=str, choices=['generate', 'register', 'docs', 'trigger'])
    parser.add_argument('--root', type=str, default=os.path.dirname(os.path.dirname(__file__)))
    
    # generate args
    parser.add_argument('--packages', type=str, default='')
    parser.add_argument('--skip-packages', type=str, default='')
    parser.add_argument('--l4t-versions', type=str, default='32.7,35.2')
    parser.add_argument('--simulate', action='store_true')
    
    # register args
    parser.add_argument('--token', type=str, default='')
    parser.add_argument('--labels', type=str, default='')
    parser.add_argument('--repo', type=str, default='https://github.com/dusty-nv/jetson-containers')
    
    args = parser.parse_args()
    
    args.packages = re.split(',|;|:', args.packages)
    args.skip_packages = re.split(',|;|:', args.skip_packages)
    args.l4t_versions = re.split(',|;|:', args.l4t_versions)
    args.labels = re.split(',|;|:', args.labels)
    
    print(args)

    packages = find_packages(args.packages, skip=args.skip_packages)
    
    if args.cmd == 'generate':
        for package in packages.values():
            for l4t_version in args.l4t_versions:
                generate_workflow(package, args.root, l4t_version, simulate=args.simulate)
    elif args.cmd == 'register':
        register_runner(args.token, args.root, args.repo, args.labels, simulate=args.simulate)