#!/usr/bin/env python3
import os
import requests
import time
from packaging.version import parse
from typing import Dict, Optional, Any

from .logging import log_warning, log_error, log_verbose

# Global session for connection pooling
_SESSION = None

def get_client() -> requests.Session:
    """
    Get or create a requests Session with connection pooling.
    """
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION

def get_package_info(package_name: str, retries: int = 5) -> Optional[Dict[str, Any]]:
    """
    Fetch package information from PyPI with retry logic and exponential backoff.

    Args:
        package_name (str): Name of the package to fetch info for
        retries (int): Number of retry attempts

    Returns:
        Optional[Dict]: Package information if successful, None otherwise
    """
    client = get_client()
    url = f"https://pypi.org/pypi/{package_name}/json"

    for attempt in range(retries):
        try:
            log_verbose(f"Fetching PyPI info for {package_name} (attempt {attempt + 1})")
            response = client.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                log_error(f"Package {package_name} not found on PyPI")
                return None

            if attempt < retries - 1:
                backoff = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                log_warning(f"Failed to fetch {package_name} from PyPI: {e}. Retrying in {backoff} seconds...")
                time.sleep(backoff)
            else:
                log_error(f"All attempts to fetch {package_name} from PyPI failed")
                return None
        except Exception as e:
            log_error(f"Unexpected error fetching {package_name} from PyPI: {e}")
            return None

def get_latest_version(package_name: str) -> Optional[str]:
    """
    Get the latest version of a package from PyPI.

    Args:
        package_name (str): Name of the package

    Returns:
        Optional[str]: Latest version string if successful, None otherwise
    """
    data = get_package_info(package_name)
    if not data or 'releases' not in data:
        return None

    # Sort versions using packaging.version.parse for proper semantic version sorting
    versions = sorted(data['releases'].keys(), key=parse)
    return versions[-1] if versions else None
