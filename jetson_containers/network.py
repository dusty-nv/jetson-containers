#!/usr/bin/env python3
import functools
import os
import requests
import time
from typing import Dict, Literal

from .logging import log_error, log_warning, log_verbose, log_info


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
            log_verbose(f"Fetching text  {url} (attempt {attempt+1})")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text.strip()
        except Exception as e:
            log_warning(f"Failed to fetch text from {url}: {e}")
            if attempt < retries - 1:
                time.sleep(backoff)
            else:
                return None


def handle_json_request(url: str, headers: Dict[str, str] = None,
                        retries: int = 3, backoff: int = 3, timeout: int = 10):
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
            log_verbose(f"Fetching json  {url} (attempt {attempt})")
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            log_error(f"HTTP {e.response.status_code} while fetching {url}")
        except requests.RequestException as e:
            log_warning(f"Request error on {url}: {e}")
        except Exception as e:
            log_warning(f"Unexpected error on {url}: {e}")

        if attempt < retries:
            time.sleep(backoff)
        else:
            raise RuntimeError(f"All {retries} attempts to fetch {url} failed.")

    return None


def get_github_token():
    """Get GitHub token from environment variables with fallbacks"""
    token = os.environ.get('GITHUB_TOKEN') or \
            os.environ.get('GITHUB_PAT') or \
            os.environ.get('GH_TOKEN')

    if not token:
        log_warning("No GitHub token found. API calls will be unauthenticated and may hit rate limits.")
        log_info("Set GITHUB_TOKEN, GITHUB_PAT, or GH_TOKEN environment variable for higher rate limits.")

    return token


@functools.lru_cache(maxsize=None)
def github_api(url: str):
    """
    Sends a request to the GitHub API using the specified URL, including authorization headers if available.

    Args:
        url (str): The GitHub API URL endpoint relative to the base URL.

    Returns:
        dict or None: The parsed JSON response data as a dictionary, or None if an error occurs.
    """
    github_token = get_github_token()
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
            log_error(f'Failed to get the value for {notation}: {e}')
            return None

    return data


def preprocess_dockerfile_for_github_api(dockerfile_path: str, pkg_path: str):
    """
    Pre-process Dockerfile to replace GitHub API calls with pre-fetched data.

    This function:
    1. Detects GitHub API calls in Dockerfiles
    2. Pre-fetches the data using authenticated API calls
    3. Creates temporary files with the fetched data
    4. Modifies the Dockerfile to use COPY instead of ADD

    Args:
        dockerfile_path (str): Path to the original Dockerfile
        pkg_path (str): Path to the package directory

    Returns:
        tuple: (modified_dockerfile_path, build_args_dict) or (original_path, None) if no changes
    """
    import re
    import json
    import os

    with open(dockerfile_path, 'r') as fp:
        content = fp.read()

    # Find all GitHub API calls
    github_api_pattern = r'ADD https://api\.github\.com/repos/([^/]+/[^/]+)/git/refs/heads/([^\s]+)\s+([^\s]+)'
    matches = re.findall(github_api_pattern, content)

    if not matches:
        return dockerfile_path, None

    # Create a temporary directory for pre-fetched data
    temp_dir = os.path.join(pkg_path, '.github-api-temp')
    os.makedirs(temp_dir, exist_ok=True)

    modified_content = content
    build_args = {}

    for owner_repo, branch, dest_path in matches:
        log_info(f"Pre-fetching GitHub data for {owner_repo}/{branch}")

        # Fetch the commit hash using authenticated API
        commit_sha = github_latest_commit(owner_repo, branch)

        if commit_sha:
            # Create a temporary file with the commit data
            temp_file = os.path.join(temp_dir, f"{owner_repo.replace('/', '_')}_{branch}.json")
            with open(temp_file, 'w') as f:
                json.dump({"sha": commit_sha, "ref": f"refs/heads/{branch}"}, f)

            # Replace ADD with COPY
            old_line = f'ADD https://api.github.com/repos/{owner_repo}/git/refs/heads/{branch} {dest_path}'
            new_line = f'COPY .github-api-temp/{os.path.basename(temp_file)} {dest_path}'
            modified_content = modified_content.replace(old_line, new_line)

            # Add build arg for the commit SHA
            build_args[f'GITHUB_{owner_repo.replace("/", "_").upper()}_COMMIT'] = commit_sha

            log_info(f"Successfully pre-fetched commit {commit_sha[:8]} for {owner_repo}/{branch}")
        else:
            log_warning(f"Failed to fetch commit for {owner_repo}/{branch}, keeping original ADD line")

    if modified_content != content:
        # Write modified Dockerfile
        modified_dockerfile = dockerfile_path + '.with-github-data'
        with open(modified_dockerfile, 'w') as fp:
            fp.write(modified_content)

        log_info(f"Created modified Dockerfile: {modified_dockerfile}")
        return modified_dockerfile, build_args

    return dockerfile_path, None
