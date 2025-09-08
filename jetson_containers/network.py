#!/usr/bin/env python3
import functools
import os
import requests
import time
from datetime import datetime
from typing import Dict, List, Literal

from .logging import log_error, log_warning, log_verbose


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


@functools.lru_cache(maxsize=None)
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
            log_error(f'Failed to get the value for {notation}: {e}')
            return None

    return data


def get_log_tail(log_file_path: str, num_lines: int = 10) -> str:
    """
    Get the last N lines from a log file.
    
    Args:
        log_file_path (str): Path to the log file
        num_lines (int): Number of lines to retrieve from the end
        
    Returns:
        str: Last N lines of the log file, or empty string if file doesn't exist
    """
    try:
        if not os.path.exists(log_file_path):
            return ""
            
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        # Get the last num_lines, remove empty lines and strip whitespace
        tail_lines = [line.strip() for line in lines[-num_lines:] if line.strip()]
        
        return '\n'.join(tail_lines)
        
    except Exception as e:
        log_verbose(f"Failed to read log file {log_file_path}: {e}")
        return ""
