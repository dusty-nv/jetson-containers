#!/usr/bin/env python3
#
# Tool for managing GitHub actions and self-hosted runners
#
# Generate build/test workflows from packages:
#   L4T_VERSION=35.2.1 python3 -m jetson_containers.ci generate
#
# Setup/register self-hosted runner service:
#   python3 -m jetson_containers.ci register --token $GITHUB_TOKEN
#
import os
import re
import yaml
import wget
import shutil
import socket
import pprint
import argparse
import subprocess

from jetson_containers import find_package, find_packages, resolve_dependencies, dependant_packages, L4T_VERSION


def generate_package_docs(package, root, repo, simulate=False):
    """
    Generate a README.md for the package
    """
    name = package['name']
    filename = os.path.join(package['path'], 'README.md')
    txt = f"# {name}\n\n"
    
    # ci/cd status
    workflows = find_package_workflows(name, root)

    if len(workflows) > 0:
        workflows = [f"[![`{workflow['name']}`]({repo}/actions/workflows/{workflow['name']}.yml/badge.svg)]({repo}/actions/workflows/{workflow['name']}.yml)" for workflow in workflows]
        txt += f"{' '.join(workflows)}\n"

    # info table
    SEP_DASH="------------"
    SEP_SPACE="            "

    txt += f"|{SEP_SPACE}|{SEP_SPACE}|\n"
    txt += f"|{SEP_DASH}|{SEP_DASH}|\n"
    
    if 'alias' in package:
        txt += f"| Aliases | { ' '.join([f'`{x}`' for x in package['alias']])} |\n"
        
    #if 'category' in package:
    #    txt += f"| Category | `{package['category']}` |\n"
                
    if 'depends' in package:
        depends = resolve_dependencies(package['depends'], check=False)
        depends = [f"[`{x}`]({find_package(x)['path'].replace(root,'')})" for x in depends]
        txt += f"| Dependencies | {' '.join(depends)} |\n"
       
    dependants = dependant_packages(name)
    
    if len(dependants) > 0:
        dependants = [f"[`{x}`]({find_package(x)['path'].replace(root,'')})" for x in dependants]
        txt += f"| Dependants | {' '.join(dependants)} |\n"
    
    #if 'dockerfile' in package:
    #    txt += f"| Dockerfile | [`{package['dockerfile']}`]({package['dockerfile']}) |\n"
        
    #if 'test' in package:
    #    txt += f"| Tests | {' '.join([f'[`{test}`]({test})' for test in package['test']])} |\n"
        
    if 'docs' in package:
        txt += f"{package['docs']}\n"
        
    if 'notes' in package:
        txt += f"{package['notes']}\n"
        
    print(filename)
    print(txt)
    
    if not simulate:
        with open(filename, 'w') as file:
            file.write(txt)
  
  
def find_package_workflows(package, root):
    """
    Find all the GitHub Workflows for building a specific package.
    """
    workflow_root = os.path.join(root, '.github/workflows')
    workflows = []
    entries = os.listdir(workflow_root)
    
    for entry in entries:
        entry_path = os.path.join(workflow_root, entry)
        
        if not os.path.isfile(entry_path):
            continue
            
        entry_ext = os.path.splitext(entry)[1]
        
        if not (entry_ext == '.yml' or entry_ext == '.yaml'):
            continue
            
        with open(entry_path) as file:
            workflow = yaml.safe_load(file)
            
        # hacky way to decipher the actual package name from the workflow,
        # since github doesn't allow custom entries and no special chars in names
        if 'run-name' not in workflow:
            continue
            
        tokens = workflow['run-name'].split(' ')
        
        if len(tokens) != 4:
            continue
            
        if package != tokens[1]:
            continue
            
        workflows.append(workflow)
        
    return workflows
        
    
def generate_workflow(package, root, l4t_version, simulate=False):
    """
    Generate the YAML workflow definition for building container for that package
    """
    name = package['name']
    workflow_name = f"{name}{'-' if ':' in name else ':'}r{l4t_version}".replace(':','_').replace('.','')
    filename = os.path.join(root, '.github/workflows', f"{workflow_name}.yml")
    
    on_paths = [
        f".github/workflows/{workflow_name}.yml",
        #"jetson_containers/**",
        os.path.join(package['path'].replace(root+'/',''), '*'),
        f"!{os.path.join(package['path'].replace(root+'/',''), 'README.md')}",
    ]

    txt = f"name: \"{workflow_name}\"\n"
    txt += f"run-name: \"Build {name} (L4T {l4t_version})\"\n"  # update find_package_workflows() if this formatting changes
    txt += "on:\n"
    txt += "  workflow_dispatch: {}\n"
    txt += "  push:\n"
    txt += "    branches:\n"
    txt += "      - 'dev'\n"
    
    if len(on_paths) > 0:
        txt += "    paths:\n"
        for on_path in on_paths:
            txt += f"      - '{on_path}'\n"  
            
    txt += "jobs:\n"
    txt += f"  {workflow_name}:\n"
    txt += f"    runs-on: [self-hosted-jetson, r{l4t_version}]\n"
    txt += "    steps:\n"
    txt += "      - uses: actions/checkout@v3\n"
    txt += f"      - run: ./build.sh --name=runner/ --build-flags='--no-cache' {package['name']}"
    
    print(filename)
    print(txt)

    if not simulate:
        with open(filename, 'w') as file:
            file.write(txt)
            

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
    
    # github runner package
    run_dir = os.path.join(root, prefix)
    run_tar = os.path.join(run_dir, "actions-runner-linux-arm64-2.304.0.tar.gz")
    run_url = "https://github.com/actions/runner/releases/download/v2.304.0/actions-runner-linux-arm64-2.304.0.tar.gz"
    
    if not os.path.isfile(run_tar) and not simulate:
        print(f"-- Installing self-hosted runner under {run_dir}")
        os.makedirs(run_dir, exist_ok=True)
        wget.download(run_url, run_tar)
        shutil.unpack_archive(run_tar, run_dir)
        
    # github cli package
    cli_deb = os.path.join(run_dir, "gh_2.32.0_linux_arm64.deb")
    cli_url = "https://github.com/cli/cli/releases/download/v2.32.0/gh_2.32.0_linux_arm64.deb"
    
    if not os.path.isfile(cli_deb) and not simulate:
        print(f"-- Downloading GitHub CLI package to {cli_deb}")
        wget.download(cli_url, cli_deb)
        cmd = f"sudo dpkg --install {cli_deb}"
        subprocess.run(cmd, executable='/bin/bash', shell=True, check=True) 
        
    # run config command
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
    
    parser.add_argument('cmd', type=str, choices=['generate', 'register', 'docs'])
    parser.add_argument('--root', type=str, default=os.path.dirname(os.path.dirname(__file__)))
    
    # generate args
    parser.add_argument('--packages', type=str, default='')
    parser.add_argument('--skip-packages', type=str, default='')
    parser.add_argument('--l4t-versions', type=str, default=str(L4T_VERSION)) #'32.7,35.2'
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
    elif args.cmd == 'docs':
        for package in packages.values():
            generate_package_docs(package, args.root, args.repo, simulate=args.simulate)
    elif args.cmd == 'register':
        register_runner(args.token, args.root, args.repo, args.labels, simulate=args.simulate)
        