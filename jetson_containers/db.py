#!/usr/bin/env python3
# Various database utilities for syncing with dockerhub, github, hf hub, ect.
import os
import re
import sys
import json
import pprint
import argparse

from jetson_containers import get_registry_containers, parse_container_versions, check_requirement
from packaging.version import Version
    

def sync_db(**kwargs):
    """
    Pull the latest metadata and export it into the graphDB.
    """
    pull_db(**kwargs)
    export_db(**kwargs)


def pull_db(user: str=None, use_cache=False, **kwargs):
    """
    Pull the latest registry metadata from DockerHub and pip.
    """
    return get_registry_containers(user=user, use_cache=use_cache, **kwargs)

def export_db(user: str=None, requires: str=None, blacklist: str=None, output: str=None, **kwargs):
    """
    Export dockerhub registry to graphDB format.
    """
    containers = pull_db(user=user, use_cache=True, **kwargs)
    nodes = {}

    for repo in containers:
        repo_name = repo['name']
        repo_node = { 'tags': ['container'] }
        repo_nodes = {}

        for container in repo['tags']:
            tag = container['name']
            tags = tag.split('-')
            key = f"{repo_name}:{tag}"
            image = f"{user}/{key}"

            if blacklist and blacklist in image:
                continue
            
            node = { 
                'tags': [repo_name],
                'docker_image': image,
                'last_modified': container['tag_last_pushed'],
                'size': container['full_size'],
                'CPU_ARCH': container['images'][0]['architecture'].replace('arm64', 'aarch64'),
            }

            node.update(parse_container_versions(image))

            if not 'L4T_VERSION' in node:
                continue

            if requires and not check_requirement(requires, 
                l4t_version=node['L4T_VERSION'], 
                cuda_version=node['CUDA_VERSION']):
                continue

            repo_nodes[key] = node

        if repo_nodes:
            nodes[repo_name] = repo_node
            nodes.update(repo_nodes)

    json_string = json.dumps(nodes, indent=2)
    print(f"\n{json_string}\n")

    if output:
        print(f"-- Saving GraphDB to:  {output} ({len(json_string)} bytes)\n")
        with open(output, 'w') as file:
            file.write(json_string)

    return nodes

if __name__ == "__main__":
    COMMANDS = {
        'sync': sync_db,
        'pull': pull_db,
        'export': export_db
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('command', type=str, choices=list(COMMANDS.keys()))
    
    #parser.add_argument('-p', '--prefer', type=str, default='local,registry,build', help="comma/colon-separated list of the source preferences (default: 'local,registry,build')")
    #parser.add_argument('-d', '--disable', type=str, default='', help="comma/colon-separated list of sources to disable (local,registry,build)")
    parser.add_argument('-u', '--user', type=str, default='dustynv', help="the DockerHub user for registry container images")
    parser.add_argument('-o', '--output', type=str, default='data/graphdb/db.json', help="file to save the selected container tag to")
    parser.add_argument('-r', '--requires', type=str, default='>=r36.4', help="limit the database export to those containers meeting these 'requires' specifiers")
    parser.add_argument('-b', '--blacklist', type=str, default='test:', help="skip the export of any containers with names that include this string")
    parser.add_argument('-q', '--quiet', action='store_true', help="use the default unattended options instead of prompting the user")
    parser.add_argument('-v', '--verbose', action='store_true', help="log extra debug/verbose info")

    args = parser.parse_args()
    args.verbose = True

    #args.prefer = re.split(',|;|:', args.prefer)
    #args.disable = re.split(',|;|:', args.disable)
    
    if args.verbose:
        os.environ['VERBOSE'] = 'ON'
    
    print(args)

    COMMANDS[args.command](**vars(args))
