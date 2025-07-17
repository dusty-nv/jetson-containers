#!/usr/bin/env python3
# Various database utilities for syncing with dockerhub, github, hf hub, ect.
import argparse
import json
import os
import pprint
import re
import sys
from datetime import datetime
from packaging.version import Version

from jetson_containers import get_registry_containers, parse_container_versions, \
    check_requirement, format_table


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

    nodes = {
        'jetson-containers': {
            'name': 'Container',
            'tags': ['container'],
        }
    }

    for repo in containers:
        repo_name = repo['name']
        repo_node = { 'tags': ['jetson-containers'] }
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

    if not output:
        return nodes

    graph_output = os.path.join(output, 'db.json')
    print(f"-- Saving GraphDB to:  {graph_output} ({len(json_string)} bytes)\n")

    with open(graph_output, 'w') as file:
        file.write(json_string)

    default_date = '2020-01-01T00:00:00.000000Z'
    #recent = sorted(nodes, key=lambda x: print(datetime.strptime(nodes[x].get('last_modified', default_date), "%Y-%m-%dT%H:%M:%S.%fZ")), reverse=True)
    recent = sorted(nodes, key=lambda x: nodes[x].get('last_modified', default_date), reverse=True)

    fields = {
        'repo': 'Repo',
        'version': 'Version',
        'docker_image': 'Image',
        #'LSB_RELEASE': 'OS',
        #'L4T_VERSION': 'L4T',
        #'CUDA_VERSION': 'CUDA',
        #'CUDA_ARCH': 'CUDA_ARCH',
        #'CPU_ARCH': 'GPU_ARCH',
        'size': 'Size (GB)',
        'last_modified': 'Timestamp',
    }

    rows = []

    def to_list(key, env):
        x = []
        for f in fields:
            if f == 'repo':
                x.append(env['tags'][0])
            elif f not in env:
                x.append('latest' if f == 'version' else '-')
            elif f == 'size':
                x.append(f"{env['size']/(1024**3):.1f}")
            elif f == 'last_modified':
                x.append(env['last_modified'][:10])
            elif f == 'docker_image':
                x.append(f"`{env['docker_image']}`")
            else:
                x.append(env[f])
        return x

    for k in recent:
        v = nodes[k]
        if len(rows) > 100:
            break
        if v.get('LSB_RELEASE') != '24.04':
            continue
        row = to_list(k,v)
        if not row:
            continue
        rows.append(row)

    table = format_table(rows, headers=[fields[x] for x in fields], tablefmt='github')
    table = table.replace('|--', '|:-').replace('--|', '-:|')

    print(f"\n{table}\n")

    table_output = os.path.join(output, 'recent.md')
    print(f"-- Saving recent containers to:  {table_output} ({len(table)} bytes)\n")

    with open(table_output, 'w') as file:
        file.write(table)

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
    parser.add_argument('-o', '--output', type=str, default='data/graphdb', help="file to save the selected container tag to")
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
