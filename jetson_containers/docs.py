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
                               resolve_dependencies, find_registry_containers, L4T_VERSION, JETPACK_VERSION)

from jetson_containers.ci import find_package_workflows, generate_workflow_badge


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
        pkg_name = os.path.basename(pkg_path)
        filename = os.path.join(pkg_path, 'README.md')
        
        txt = ''
        docs = ''

        for name, package in pkgs.items():
            # rolldown for subpackages
            if len(pkgs) > 1:
                txt += "<details open>\n"
                txt += f"<summary><h3>{name}</h3></summary>\n\n"
            
            # info table
            txt += f"|{_TABLE_SPACE}|{_TABLE_SPACE}|\n"
            txt += f"|{_TABLE_DASH}|{_TABLE_DASH}|\n"
            
            if 'alias' in package:
                txt += f"| Aliases | { ' '.join([f'`{x}`' for x in package['alias']])} |\n"
                
            # ci/cd status
            workflows = find_package_workflows(name, root)

            if len(workflows) > 0:
                workflows = [generate_workflow_badge(workflow, repo) for workflow in workflows]
                txt += f"| Builds | {' '.join(workflows)} |\n"
                
            #if 'category' in package:
            #    txt += f"| Category | `{package['category']}` |\n"
                 
            txt += f"| Requires | `L4T {package['requires']}` |\n"
            
            if 'depends' in package:
                depends = resolve_dependencies(package['depends'], check=False)
                depends = [f"[`{x}`]({find_package(x)['path'].replace(root,'')})" for x in depends]
                txt += f"| Dependencies | {' '.join(depends)} |\n"
               
            dependants = dependant_packages(name)
            
            if len(dependants) > 0:
                dependants = [f"[`{x}`]({find_package(x)['path'].replace(root,'')})" for x in sorted(dependants)]
                txt += f"| Dependants | {' '.join(dependants)} |\n"
            
            if 'dockerfile' in package:
                txt += f"| Dockerfile | [`{package['dockerfile']}`]({package['dockerfile']}) |\n"
                
            #if 'test' in package:
            #    txt += f"| Tests | {' '.join([f'[`{test}`]({test})' for test in package['test']])} |\n"
            
            if 'notes' in package:
                txt += f"| Notes | {package['notes']} |\n"
                
            if 'docs' in package:
                docs = package['docs']
                
            if len(pkgs) > 1:
                txt += "</details>\n"
        
        # add the help text back to the top (if one of the packages had it)
        if docs:
            txt = docs + '\n' + txt
            
        txt = f"# {pkg_name}\n\n" + txt
        
        # example commands for running the container
        run_txt = "### Run Container\n"
        run_txt += "[`run.sh`](/run.sh) adds some default `docker run` args (like `--runtime nvidia`, mounts a [`/data`](/data) cache, and detects devices)\n" 
        run_txt += "```bash\n"
        run_txt += "# automatically pull or build a compatible container image\n"
        run_txt += f"./run.sh $(./autotag {pkg_name})\n"
        run_img = f"{pkg_name}:{L4T_VERSION}\n"
        
        # list all the dockerhub images for this group of packages
        registry = find_registry_containers(pkg_name, check_l4t_version=False, return_dicts=True)
        
        if len(registry) > 0:
            pprint.pprint(registry)
            run_txt += "\n# or manually specify one of the container images above\n"
            run_img = f"{registry[0]['namespace']}/{registry[0]['name']}:{registry[0]['tags'][0]['name']}"
            run_txt += f"./run.sh {run_img}\n"
            txt += "### Container Images\n"
            for container in registry:
                for tag in container['tags']:
                    txt += f"- [`{container['namespace']}/{container['name']}:{tag['name']}`](https://hub.docker.com/r/{container['namespace']}/{container['name']}/tags)  `{tag['images'][0]['architecture']}`  `({tag['full_size']/(1024**3):.1f}GB)`\n"
            txt += "\n"
        
        run_txt += "\n# or if using 'docker run' (specify image and mounts/ect)\n"
        run_txt += f"sudo docker run --runtime nvidia -it --rm --network=host {run_img}\n"
        run_txt += "```\n"
        run_txt += f"To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:\n"
        run_txt += "```bash\n"
        run_txt += f"./run.sh -v /path/on/host:/path/in/container $(./autotag {pkg_name})\n"
        run_txt += "```\n"
        run_txt += f"To start the container running a command, as opposed to the shell:\n"
        run_txt += "```bash\n"
        run_txt += f"./run.sh $(./autotag {pkg_name}) my_app --abc xyz\n"
        run_txt += "```\n"
        
        run_txt += "### Build Container\n"
        run_txt += "If you use [`autotag`](/autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do this System Setup, then run:\n"
        run_txt += "```bash\n"
        run_txt += f"./build.sh {pkg_name}\n"
        run_txt += "```\n"
        run_txt += "The dependencies from above will be built into the container, and it'll be tested.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.\n"
        
        txt += run_txt
        
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

        