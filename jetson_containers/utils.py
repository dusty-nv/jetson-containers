#!/usr/bin/env python3
import os
import grp
import sys
import pprint
import requests, time
import functools

from typing import Dict, Literal


def check_dependencies(install: bool = True):
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


def needs_sudo(group: str = 'docker') -> bool:
    """
    Returns true if sudo is needed to use the docker engine (if user isn't in the docker group)
    """
    if is_root_user():
        return False
    else:
        return not user_in_group(group)


def sudo_prefix(group: str = 'docker'):
    """
    Returns a sudo prefix for command strings if the user needs sudo for accessing docker
    """
    if needs_sudo(group):
        #print('USER NEEDS SUDO')
        return "sudo "
    else:
        return ""


def handle_text_request(url, retries=3, backoff=5):
    """
    Handles a request to fetch text data from the given URL.

    Args:
        url (str): The URL from which to fetch text data.

    Returns:
        str or None: The fetched text data, stripped of leading and trailing whitespace,
                     or None if an error occurs.
    """
    for attempt in range(retries):
        try:
            print(f"[INFO] Fetching: {url} (attempt {attempt+1})")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text.strip()
        except Exception as e:
            print(f"[WARN] Failed to fetch {url}: {e}")
            if attempt < retries - 1:
                time.sleep(backoff)
            else:
                return None


def handle_json_request(url: str, headers: Dict[str, str] = None, retries: int = 3, backoff: int = 3, timeout: int = 10):
    """
    Fetch JSON data from a URL with retry, timeout, and backoff handling.

    Args:
        url (str): The URL to fetch.
        headers (dict): Optional HTTP headers.
        retries (int): Number of retry attempts.
        backoff (int): Seconds to wait between retries.
        timeout (int): Timeout in seconds for each request.

    Returns:
        dict or None: Parsed JSON response, or None if all attempts fail.
    """
    for attempt in range(1, retries + 1):
        try:
            print(f"[info] GET {url} (attempt {attempt})")
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            print(f"[error] HTTP {e.response.status_code} while fetching {url}")
        except requests.RequestException as e:
            print(f"[warn] Request error on {url}: {e}")
        except Exception as e:
            print(f"[warn] Unexpected error on {url}: {e}")

        if attempt < retries:
            time.sleep(backoff)
        else:
            print(f"[fail] All attempts to fetch {url} failed.")

    return None

@functools.cache
def github_api(url: str):
    """
    Sends a request to the GitHub API using the specified URL, including authorization headers if available.

    Args:
        url (str): The GitHub API URL endpoint relative to the base URL.

    Returns:
        dict or None: The parsed JSON response data as a dictionary, or None if an error occurs.
    """
    github_token = os.environ.get('GITHUB_TOKEN')
    headers = {'Authorization': f'token {github_token}'} if github_token else None
    request_url = f'https://api.github.com/{url}'

    return handle_json_request(request_url, headers)


def github_latest_commit(repo: str, branch: str = 'main'):
    """
    Retrieves the latest commit SHA from the specified branch of a GitHub repository.

    Args:
        repo (str): The full name of the GitHub repository in the format 'owner/repo'.
        branch (str, optional): The branch name. Defaults to 'main'.

    Returns:
        str or None: The SHA (hash) of the latest commit, or None if no commit is found.
    """
    commit_info = github_api(f"repos/{repo}/commits/{branch}")
    return commit_info.get('sha') if commit_info else None


def github_latest_tag(repo: str):
    """
    Retrieves the latest tag name from the specified GitHub repository.

    Args:
        repo (str): The full name of the GitHub repository in the format 'owner/repo'.

    Returns:
        str or None: The name of the latest tag, or None if no tags are found.
    """
    tags = github_api(f"repos/{repo}/tags")
    return tags[0].get('name') if tags else None


def get_json_value_from_url(url: str, notation: str = None):
    """
    Retrieves JSON data from the given URL and returns either the whole data or a specified nested value using a dot notation string.

    Args:
        url (str): The URL from which to fetch the JSON data.
        notation (str, optional): A dot notation string specifying the nested property to retrieve.
                                  If None or an empty string is provided, the entire JSON data is returned.

    Returns:
        str or dict: The value of the specified nested property or the whole data if `notation` is None.
                     Returns None if the specified property does not exist.
    """
    data = handle_json_request(url)

    if notation and data is not None:
        keys = notation.split('.') if '.' in notation else [notation]
        current = data

        try:
            for key in keys:
                current = current[key]
            return str(current).strip()
        except KeyError as e:
            print(f'ERROR: Failed to get the value for {notation}: {e}')
            return None

    return data


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


def cprint(text, color):
    """
    Print string to terminal in the specified color.  The recognized colors are found below.
    """
    print(colorized(text, color))


def colorized(text, color):
    """
    Return string encased with ANSI terminal color codes.  The recognized colors are found below.
    """
    colors = {
        'red': 91,
        'green': 92,
        'yellow': 93,
        'magenta': 94,
        'purple': 95,
        'cyan': 96,
        'grey': 97,
        'black': 98,
    }

    color = color.lower()

    if color not in colors:
        return text

    return f"\033[{colors[color]}m{text}\033[00m"


def format_table(rows, header=None, footer=None, color='green', attrs=None, tablefmt='simple_grid', **kwargs):
    """
    Print a table from a list[list] of rows/columns, or a 2-column dict
    where the keys are column 1, and the values are column 2.

    Header is a list of columns or rows that are inserted at the top.
    Footer is a list of columns or rows that are added to the end.

    color names and style attributes are from termcolor library:
      https://github.com/termcolor/termcolor#text-properties
    """
    from tabulate import tabulate
    #from termcolor import cprint, colored

    kwargs.setdefault('numalign', 'center')

    def limit_str(text, max_len):
        if not isinstance(text, str):
            return text

        if len(text) > max_len:
            return text[:max_len]
        else:
            return text

    if isinstance(rows, dict):
        rows = [[limit_str(key,35), limit_str(value, 75)] for key, value in rows.items() if not isinstance(value, dict)]

    if header:
        if not isinstance(header[0], list):
            header = [header]
        rows = header + rows

    if footer:
        if not isinstance(footer[0], list):
            footer = [footer]
        rows = rows + footer

    #cprint(tabulate(rows, tablefmt='simple_grid', numalign='center'), color, attrs=attrs)
    return tabulate(rows, tablefmt=tablefmt, **kwargs)