from jetson_containers import L4T_VERSION, github_latest_tag, log_warning, log_verbose
from jetson_containers.pypi_utils import get_latest_version
from packaging.version import parse as parse_version
import os
import json
import time
import random
import requests
from typing import Optional
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError
from urllib3.exceptions import MaxRetryError, ProtocolError

# Default versions as fallback
DEFAULT_VERSIONS = {
    'transformers': '4.55.3',  # Latest known working version
}

# Cache file path
CACHE_DIR = os.path.expanduser('~/.cache/jetson-containers')
CACHE_FILE = os.path.join(CACHE_DIR, 'pypi_versions.json')
CACHE_DURATION = 3600  # 1 hour in seconds


def ensure_cache_dir():
    """Ensure the cache directory exists."""
    if not os.path.exists(CACHE_DIR):
        log_verbose(f"Creating cache directory: {CACHE_DIR}")
        os.makedirs(CACHE_DIR, exist_ok=True)


def load_cached_versions() -> Optional[dict]:
    """Load versions from cache if they exist and are not expired."""
    try:
        if not os.path.exists(CACHE_FILE):
            log_verbose(f"Cache file does not exist: {CACHE_FILE}")
            return None

        # Check if cache is expired
        cache_age = time.time() - os.path.getmtime(CACHE_FILE)
        if cache_age > CACHE_DURATION:
            log_verbose(f"Cache expired (age: {cache_age:.1f}s > {CACHE_DURATION}s)")
            return None

        with open(CACHE_FILE, 'r') as f:
            versions = json.load(f)
            log_verbose(f"Loaded versions from cache: {versions}")
            return versions
    except Exception as e:
        log_warning(f"Failed to load cached versions: {str(e)}")
        return None


def save_versions_to_cache(versions: dict):
    """Save versions to cache file."""
    try:
        ensure_cache_dir()
        with open(CACHE_FILE, 'w') as f:
            json.dump(versions, f)
        log_warning(f"Saved versions to cache: {versions}")
    except Exception as e:
        log_warning(f"Failed to save versions to cache: {str(e)}")


def fetch_pypi_with_retry(package_name: str, max_retries: int = 5, initial_delay: int = 5) -> Optional[dict]:
    """
    Fetch package info from PyPI with retry logic and caching.

    Args:
        package_name: Name of the package to fetch
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds

    Returns:
        Package info as dict or None if all retries fail
    """
    # Try cache first
    cached_versions = load_cached_versions()
    if cached_versions and package_name in cached_versions:
        log_verbose(f"Using cached version for {package_name}: {cached_versions[package_name]}")
        return {'releases': {cached_versions[package_name]: []}}

    delay = initial_delay
    url = f"https://pypi.org/pypi/{package_name}/json"

    for attempt in range(max_retries):
        try:
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0.5, 1.5)
            timeout = 15 + (attempt * 5)  # Increased base timeout

            # Create a new session for each attempt
            session = requests.Session()

            # Add headers to help with connection stability
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; jetson-containers/1.0)',
                'Accept': 'application/json',
                'Connection': 'keep-alive',
                'Accept-Encoding': 'gzip, deflate',
            }

            # Configure session for better reliability
            session.mount('https://', requests.adapters.HTTPAdapter(
                max_retries=3,
                pool_connections=1,
                pool_maxsize=1
            ))

            log_verbose(f"Attempting to fetch {url} (attempt {attempt + 1}/{max_retries})")
            response = session.get(url, timeout=timeout, headers=headers, verify=True)
            response.raise_for_status()
            data = response.json()

            # Save to cache if we got valid data
            if data and 'releases' in data:
                sorted_versions = sorted(data['releases'].keys(), key=parse_version)
                version = sorted_versions[-1]
                log_verbose(f"Successfully fetched version {version} from PyPI")
                versions = {package_name: version}
                save_versions_to_cache(versions)

            return data

        except requests.exceptions.SSLError as e:
            log_warning(f"SSL error on {url}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay * jitter)
                delay *= 2
            else:
                log_warning(f"All attempts to fetch {url} failed due to SSL errors.")
                return None
        except (requests.exceptions.RequestException, Timeout, ConnectionError, HTTPError, MaxRetryError, ProtocolError) as e:
            if attempt < max_retries - 1:
                log_warning(f"Request error on {url}: {str(e)}")
                time.sleep(delay * jitter)
                delay *= 2
            else:
                log_warning(f"All attempts to fetch {url} failed.")
                return None
        finally:
            if 'session' in locals():
                session.close()


def transformers_pypi(version, default=False, requires=None) -> list:
    pkg = package.copy()

    if version == 'latest':
        log_verbose(f"Fetching latest version for transformers_pypi (default={default}, requires={requires})")
        # Try cache first
        cached_versions = load_cached_versions()
        if cached_versions and 'transformers' in cached_versions:
            version = cached_versions['transformers']
            log_verbose(f"Using cached version: {version}")
        else:
            data = fetch_pypi_with_retry('transformers')
            if data is None or 'releases' not in data:
                log_warning(f"Using default version {DEFAULT_VERSIONS['transformers']} due to network failure")
                version = DEFAULT_VERSIONS['transformers']
            else:
                sorted_versions = sorted(data['releases'].keys(), key=parse_version)
                version = sorted_versions[-1]
                log_verbose(f"Using latest version from PyPI: {version}")

    pkg['name'] = f'transformers:{version}'

    if default:
        pkg['alias'] = 'transformers'

    if requires:
        pkg['requires'] = requires

    return pkg


def transformers_git(version, repo='huggingface/transformers', branch=None, requires=None, default=False) -> list:
    """Create a package for transformers from git repository."""
    pkg = package.copy()

    if version == 'latest':
        version = github_latest_tag(repo)
        log_verbose(f"Using latest git tag: {version}")

    pkg['name'] = f'transformers:{version}'
    pkg['build_args'] = {
        'TRANSFORMERS_PACKAGE': f'git+https://github.com/{repo}.git@{version}',
        'TRANSFORMERS_VERSION': version
    }

    if default:
        pkg['alias'] = 'transformers'
        log_warning(f"Set git package alias to: {pkg['alias']}")

    if requires:
        pkg['requires'] = requires
        log_warning(f"Set git package requires to: {pkg['requires']}")

    return pkg


# 11/3/23 - removing 'bitsandbytes' and 'auto_gptq' due to circular dependency and increased load times of
# anything using transformers if you want to use load_in_8bit/load_in_4bit or AutoGPTQ quantization
# built-into transformers, use the 'bitsandbytes' or 'auto_gptq' containers directly instead of transformers container
package = [
    transformers_pypi('latest', default=(L4T_VERSION.major >= 36), requires='>=36'),
    transformers_pypi('4.46.3', default=(L4T_VERSION.major < 36), requires='<36'),   # 4.46.3 is the last version that supports Python 3.8
    # Commenting out git version to avoid version conflicts
    # transformers_git('latest', default=False, requires=None),                         # will always resolve to the latest git version from huggingface/transformers
]
