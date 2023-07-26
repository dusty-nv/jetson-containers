#!/usr/bin/env python3
#
# Generate one or more package's README.md:
#   python3 -m jetson_containers.docs package pytorch tensorflow
#   python3 -m jetson_containers.docs package *
#
# Generate the package index (packages/README.md)
#   python3 -m jetson_containers.docs index
#
import os
import re
import pprint
import argparse

from jetson_containers import (find_package, find_packages, group_packages, dependant_packages, 
                               resolve_dependencies, L4T_VERSION, JETPACK_VERSION)

from jetson_containers.ci import find_package_workflows


_TABLE_DASH="------------"
_TABLE_SPACE="            "
    
    
def generate_package_list(packages, root, repo, filename='packages/README.md', simulate=False):
    """
    Generate a markdown table of all the packages
    """
    filename = os.path.join(root, filename)
    
    txt = "# Packages\n"
    txt += f"|{_TABLE_SPACE}|{_TABLE_SPACE}|\n"
    txt += f"|{_TABLE_DASH}|{_TABLE_DASH}|\n"
        
    # group packages by category for navigability
    groups = group_packages(packages, key='group', default='other')
    
    for group_name in sorted(list(groups.keys())):
        group = groups[group_name]
        
        txt += f"| **`{group_name.upper()}`** | |\n"
        
        for name in sorted(list(group.keys())):
            package = group[name]
            txt += f"| &nbsp;&nbsp; [`{name}`]({package['path'].replace(root,'')}) | "
            
            workflows = find_package_workflows(name, root)

            if len(workflows) > 0:
                workflows = [f"[![`{workflow['name']}`]({repo}/actions/workflows/{workflow['name']}.yml/badge.svg)]({repo}/actions/workflows/{workflow['name']}.yml)" for workflow in workflows]
                txt += f"{' '.join(workflows)}"

            txt += " |\n"
        
    print(filename)
    print(txt)
    
    if not simulate:
        with open(filename, 'w') as file:
            file.write(txt)
    
    
def generate_package_docs(packages, root, repo, simulate=False):
    """
    Generate README.md files for the supplied packages.
    Group them by path so there's just one page per directory.
    """
    groups = group_packages(packages, 'path')
    
    for pkg_path, pkgs in groups.items():
        filename = os.path.join(pkg_path, 'README.md')
        txt = f"# {os.path.basename(pkg_path)}\n\n"
        
        docs = ''
        notes = ''
        
        for name, package in pkgs.items():
        
            if len(pkgs) > 1:
                txt += "<details open>\n"
                txt += f"<summary>{name}</summary>\n\n"

            # ci/cd status
            workflows = find_package_workflows(name, root)

            if len(workflows) > 0:
                workflows = [f"[![`{workflow['name']}`]({repo}/actions/workflows/{workflow['name']}.yml/badge.svg)]({repo}/actions/workflows/{workflow['name']}.yml)" for workflow in workflows]
                txt += f"{' '.join(workflows)}\n"
            
            # info table
            txt += f"|{_TABLE_SPACE}|{_TABLE_SPACE}|\n"
            txt += f"|{_TABLE_DASH}|{_TABLE_DASH}|\n"
            
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
                dependants = [f"[`{x}`]({find_package(x)['path'].replace(root,'')})" for x in sorted(dependants)]
                txt += f"| Dependants | {' '.join(dependants)} |\n"
            
            #if 'dockerfile' in package:
            #    txt += f"| Dockerfile | [`{package['dockerfile']}`]({package['dockerfile']}) |\n"
                
            #if 'test' in package:
            #    txt += f"| Tests | {' '.join([f'[`{test}`]({test})' for test in package['test']])} |\n"
                
            if 'docs' in package and package['docs'] != docs:
                txt += f"\n{package['docs']}\n"
                docs = package['docs']
                
            if 'notes' in package and package['notes'] != notes:
                txt += f"\n{package['notes']}\n"
                notes = package['notes']
        
            if len(pkgs) > 1:
                txt += "</details>\n"
        
        print(filename)
        print(txt)
    
        if not simulate:
            with open(filename, 'w') as file:
                file.write(txt)
            
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
    txt += f"|{_TABLE_SPACE}|{_TABLE_SPACE}|\n"
    txt += f"|{_TABLE_DASH}|{_TABLE_DASH}|\n"
    
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
        dependants = [f"[`{x}`]({find_package(x)['path'].replace(root,'')})" for x in sorted(dependants)]
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
    """
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('cmd', type=str, choices=['package', 'packages', 'index'])
    parser.add_argument('packages', type=str, nargs='*', default=[], help='packages to generate docs for')
    
    parser.add_argument('--root', type=str, default=os.path.dirname(os.path.dirname(__file__)))
    parser.add_argument('--repo', type=str, default='https://github.com/dusty-nv/jetson-containers')
    parser.add_argument('--skip-packages', type=str, default='')
    parser.add_argument('--simulate', action='store_true')
    
    args = parser.parse_args()
    args.skip_packages = re.split(',|;|:', args.skip_packages)
    
    print(args)

    packages = find_packages(args.packages, skip=args.skip_packages)
    
    if args.cmd == 'package' or args.cmd == 'packages':
        generate_package_docs(packages, args.root, args.repo, simulate=args.simulate)
    elif args.cmd == 'index':
        generate_package_list(packages, args.root, args.repo, simulate=args.simulate)

        