#!/usr/bin/env python3
#
# Generate one or more package's README.md:
#   python3 -m jetson_containers.docs package pytorch tensorflow
#   python3 -m jetson_containers.docs package *
#
# Generate the package index (packages/README.md)
#   python3 -m jetson_containers.docs index
#
import argparse
import dockerhub_api
import os
import pprint
import re
import time

from jetson_containers import (find_package, find_packages, group_packages,
                               dependant_packages, package_scan_options,
                               resolve_dependencies, find_registry_containers,
                               L4T_VERSION, JETPACK_VERSION)
from jetson_containers.ci import find_package_workflows, generate_workflow_badge
from jetson_containers.utils import split_container_name

_TABLE_DASH = "------------"
_TABLE_SPACE = "            "
_NBSP = "&nbsp;&nbsp;&nbsp;"


def is_builder(name) -> bool:
    return (
        name.endswith('-builder')
        or name.endswith(':builder')
    )


def generate_package_list(packages, root, repo, filename='packages/README.md',
                          simulate=False):
    """
    Generate a markdown table of all the packages
    """
    filename = os.path.join(root, filename)

    # group packages by category for navigability
    groups = group_packages(packages, key='group', default='other')

    txt = "# Packages\n"
    txt += "> "

    # build list of groups
    for group_name in sorted(list(groups.keys())):
        txt += f"[`{group_name.upper()}`](#user-content-{group_name}) "
        # txt += f"* [**`{group_name.upper()}`**](#user-content-{group_name})\n"

    txt += "\n\n"

    # build package table
    txt += f"|{_TABLE_SPACE}|{_TABLE_SPACE}|\n"
    txt += f"|{_TABLE_DASH}|{_TABLE_DASH}|\n"

    for group_name in sorted(list(groups.keys())):
        group = groups[group_name]

        txt += f'| <a id="{group_name}">**`{group_name.upper()}`**</a> | |\n'

        for name in sorted(list(group.keys())):
            package = group[name]
            txt += f"| {_NBSP} [`{name}`]({package['path'].replace(root, '')}) | "

            workflows = find_package_workflows(name, root)

            if len(workflows) > 0:
                workflows = [generate_workflow_badge(workflow, repo) for workflow in
                             sorted(workflows, key=lambda x: x['name'])]
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
    groups = groups.items()
    total_groups = len(groups)

    for groupd_idx, pkg_data in enumerate(groups):
        pkg_path, pkgs = pkg_data
        pkg_name = os.path.basename(pkg_path)
        filename = os.path.join(pkg_path, 'README.md')

        unique_pkgs = {k: v for k, v in pkgs.items() if not is_builder(k)}
        total_pkgs = len(unique_pkgs)

        print(
            f" 📝 [{groupd_idx + 1}/{total_groups}] Generating docs for {pkg_name} with {total_pkgs} versions:")

        txt = f"# {pkg_name}\n\n"
        docs = ""

        txt += "> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)\n\n"

        for package in pkgs.values():
            if 'docs' in package:
                docs = package['docs']
                break

        if docs:
            docs_path = os.path.join(pkg_path, docs)

            if os.path.isfile(docs_path):
                with open(docs_path) as file:
                    docs = file.read()
            # else:
            #    txt = docs + "\n"
            #    docs = ""

            txt += docs + "\n"

        txt += "<details open>\n"
        txt += '<summary><b><a id="containers">CONTAINERS</a></b></summary>\n<br>\n\n'

        for i, name, in enumerate(unique_pkgs):
            package = unique_pkgs[name]

            print(f"\t- [{i + 1}/{total_pkgs} versions] {name}...")

            txt += f"| **`{name}`** | |\n"
            txt += f"| :-- | :-- |\n"

            if 'alias' in package:
                txt += f"| {_NBSP}Aliases | {' '.join([f'`{x}`' for x in package['alias']])} |\n"

            # ci/cd status
            workflows = find_package_workflows(name, root)

            if len(workflows) > 0:
                workflows = [generate_workflow_badge(workflow, repo) for workflow in
                             workflows]
                txt += f"| {_NBSP}Builds | {' '.join(workflows)} |\n"

            # if 'category' in package:
            #    txt += f"| Category | `{package['category']}` |\n"

            txt += f"| {_NBSP}Requires | `L4T {package['requires']}` |\n"

            if 'depends' in package:
                depends = resolve_dependencies(package['depends'], check=False)
                depends = [f"[`{x}`]({find_package(x)['path'].replace(root, '')})" for x
                           in depends]
                txt += f"| {_NBSP}Dependencies | {' '.join(depends)} |\n"

            dependants = dependant_packages(name)

            if len(dependants) > 0:
                dependants_links = []
                for dependant in sorted(dependants):
                    dep_pkg = find_package(dependant)

                    if not isinstance(dep_pkg, dict):
                        continue

                    dep_path = dep_pkg.get('path')
                    dep_name = dep_pkg.get('name')

                    if isinstance(dep_path, str) and not is_builder(dep_name):
                        dependants_links.append(
                            f"[`{dependant}`]({dep_path.replace(root, '')})")

                if dependants_links:
                    txt += f"| {_NBSP}Dependants | {' '.join(dependants_links)} |\n"

            if 'dockerfile' in package:
                txt += f"| {_NBSP}Dockerfile | [`{package['dockerfile']}`]({package['dockerfile']}) |\n"

            # if 'test' in package:
            #    txt += f"| Tests | {' '.join([f'[`{test}`]({test})' for test in package['test']])} |\n"

            # list all the dockerhub images for this specific package
            registry = find_registry_containers(name, check_l4t_version=False,
                                                return_dicts=True)

            if len(registry) > 0:
                reg_txt = []

                for container in registry:
                    for tag in sorted(container['tags'], key=lambda x: x[
                        'name']):  # x['tag_last_pushed'], reverse=True):
                        reg_txt.append(
                            f"[`{container['namespace']}/{container['name']}:{tag['name']}`](https://hub.docker.com/r/{container['namespace']}/{container['name']}/tags) `({tag['tag_last_pushed'][:10]}, {tag['full_size'] / (1024 ** 3):.1f}GB)`")

                txt += f"| {_NBSP}Images | {'<br>'.join(reg_txt)} |\n"

            if 'notes' in package:
                txt += f"| {_NBSP}Notes | {package['notes']} |\n"

            txt += "\n"

        txt += "</details>\n"

        # example commands for running the container
        run_txt = "\n<details open>\n"
        run_txt += '<summary><b><a id="run">RUN CONTAINER</a></b></summary>\n<br>\n\n'
        run_txt += "To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:\n"
        run_txt += "```bash\n"
        run_txt += "# automatically pull or build a compatible container image\n"
        run_txt += f"jetson-containers run $(autotag {pkg_name})\n"
        run_img = f"{pkg_name}:{L4T_VERSION}\n"

        # list all the dockerhub images for this group of packages
        registry = find_registry_containers(pkg_name, check_l4t_version=False,
                                            return_dicts=True)

        if len(registry) > 0:
            # pprint.pprint(registry)

            run_txt += "\n# or explicitly specify one of the container images above\n"
            run_img = f"{registry[0]['namespace']}/{registry[0]['name']}:{registry[0]['tags'][0]['name']}"
            run_txt += f"jetson-containers run {run_img}\n"

            txt += "\n<details open>\n"
            txt += '<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>\n<br>\n\n'
            txt += "| Repository/Tag | Date | Arch | Size |\n"
            txt += "| :-- | :--: | :--: | :--: |\n"

            for container in registry:
                for tag in sorted(container['tags'], key=lambda x: x[
                    'name']):  # x['tag_last_pushed'], reverse=True):
                    txt += f"| &nbsp;&nbsp;[`{container['namespace']}/{container['name']}:{tag['name']}`](https://hub.docker.com/r/{container['namespace']}/{container['name']}/tags) "
                    txt += f"| `{tag['tag_last_pushed'][:10]}` "
                    txt += f"| `{tag['images'][0]['architecture']}` "
                    txt += f"| `{tag['full_size'] / (1024 ** 3):.1f}GB` |\n"

            txt += "\n"
            txt += "> <sub>Container images are compatible with other minor versions of JetPack/L4T:</sub><br>\n"
            txt += "> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R32.7 containers can run on other versions of L4T R32.7 (JetPack 4.6+)</sub><br>\n"
            txt += "> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R35.x containers can run on other versions of L4T R35.x (JetPack 5.1+)</sub><br>\n"
            txt += "</details>\n"

        run_txt += "\n# or if using 'docker run' (specify image and mounts/ect)\n"
        run_txt += f"sudo docker run --runtime nvidia -it --rm --network=host {run_img}\n"
        run_txt += "```\n"
        run_txt += "> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>\n"
        run_txt += "> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>\n\n"

        run_txt += f"To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:\n"
        run_txt += "```bash\n"
        run_txt += f"jetson-containers run -v /path/on/host:/path/in/container $(autotag {pkg_name})\n"
        run_txt += "```\n"
        run_txt += f"To launch the container running a command, as opposed to an interactive shell:\n"
        run_txt += "```bash\n"
        run_txt += f"jetson-containers run $(autotag {pkg_name}) my_app --abc xyz\n"
        run_txt += "```\n"
        run_txt += "You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.\n"
        run_txt += "</details>\n"

        run_txt += "<details open>\n"
        run_txt += '<summary><b><a id="build">BUILD CONTAINER</b></summary>\n<br>\n\n'
        run_txt += "If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:\n"
        run_txt += "```bash\n"
        run_txt += f"jetson-containers build {pkg_name}\n"
        run_txt += "```\n"
        run_txt += "The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.\n"
        run_txt += "</details>\n"

        # if docs:
        #    txt += "\n<details open>\n"
        #    txt += "<summary><b>CONTAINER DOCS</b></summary>\n<br>\n\n"
        #    txt += docs + "\n"
        #    txt += "</details>\n"

        txt += run_txt

        # print(filename)
        # print(txt)

        if not simulate:
            with open(filename, 'w') as file:
                file.write(txt)


def generate_registry_docs(packages, root, repo, user, password, simulate=False):
    """
    Apply descriptions to the container repos on DockerHub
    """
    groups = group_packages(packages, 'path')
    hub = dockerhub_api.DockerHub(username=user, password=password, return_lists=True)
    request_cache = []

    for name, package in packages.items():
        namespace, repository, tag = split_container_name(name)
        repo_path = package['path'][package['path'].find('/packages/') + 1:]
        readme_path = os.path.join(package['path'], 'README.md')

        if repository in request_cache:
            continue

        request_cache.append(repository)

        short = f"{repo}/{repo_path}"

        with open(readme_path, 'r') as file:
            full = file.read()

        full = full.replace("](/", f"]({repo}/tree/master/")
        full = full.replace("](Dockerfile",
                            f"]({repo}/tree/master/{repo_path}/Dockerfile")
        full = full[:24999] if len(full) >= 25000 else full  # length limit

        print(f"-- Setting DockerHub description for {user}/{repository}")
        print(f"        {short}")
        print(f"        {readme_path}")
        print(full)

        if not simulate:
            try:
                hub.set_repository_description(user, repository, descriptions={
                    'short': short,
                    'full': full,
                })
            except Exception as err:
                print(
                    f"Exception occurred setting DockerHub container readme for {name}\n",
                    err)

        # if not simulate:
        #    time.sleep(5.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('cmd', type=str,
                        choices=['package', 'packages', 'index', 'registry'])
    parser.add_argument('packages', type=str, nargs='*', default=[],
                        help='packages to generate docs for')

    parser.add_argument('--root', type=str,
                        default=os.path.dirname(os.path.dirname(__file__)))
    parser.add_argument('--repo', type=str,
                        default='https://github.com/dusty-nv/jetson-containers')
    parser.add_argument('--user', type=str, default='dustynv',
                        help="the DockerHub user for registry container images")
    parser.add_argument('--password', type=str, default='',
                        help="DockerHub password (only needed for 'registry' command)")
    parser.add_argument('--skip-packages', type=str, default='')
    parser.add_argument('--skip-l4t-checks', action='store_true')
    parser.add_argument('--simulate', action='store_true')

    args = parser.parse_args()
    args.skip_packages = re.split(',|;|:', args.skip_packages)

    print(args)

    if args.skip_l4t_checks:
        package_scan_options({'check_l4t_version': False})

    packages = find_packages(args.packages, skip=args.skip_packages)

    if args.cmd == 'package' or args.cmd == 'packages':
        generate_package_docs(packages, args.root, args.repo, simulate=args.simulate)
    elif args.cmd == 'index':
        generate_package_list(packages, args.root, args.repo, simulate=args.simulate)
    elif args.cmd == 'registry':
        generate_registry_docs(packages, args.root, args.repo, args.user, args.password,
                               simulate=args.simulate)
