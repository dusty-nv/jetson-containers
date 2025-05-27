import requests
import yaml
import time
import random
import os
import json
from typing import Tuple, Optional
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError
from urllib3.exceptions import MaxRetryError, ProtocolError

from jetson_containers import github_latest_tag, log_warning, log_verbose

# Default versions as fallback
DEFAULT_VERSIONS = {
    'BASHIO_VERSION': '0.16.2',
    'TEMPIO_VERSION': '2024.11.2',
    'S6_OVERLAY_VERSION': '3.1.6.2'
}

# Cache file path
CACHE_DIR = os.path.expanduser('~/.cache/jetson-containers')
CACHE_FILE = os.path.join(CACHE_DIR, 'homeassistant_versions.json')
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
        log_verbose(f"Saved versions to cache: {versions}")
    except Exception as e:
        log_warning(f"Failed to save versions to cache: {str(e)}")


def fetch_with_retry(url: str, max_retries: int = 5, initial_delay: int = 5) -> Optional[str]:
    """
    Fetch content with improved retry logic, exponential backoff, and jitter.

    Args:
        url: The URL to fetch
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds

    Returns:
        The content as string or None if all retries fail
    """
    delay = initial_delay
    url = f"https://raw.githubusercontent.com/home-assistant/docker-base/master/ubuntu/build.yaml"

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
                'Accept': 'application/json, text/plain, */*',
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
            return response.text

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


def latest_deps_versions(branch_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get the latest dependency versions from the Home Assistant docker-base repository.
    Uses cached values if available and not expired, falls back to defaults if all else fails.

    Args:
        branch_name: The branch/tag to fetch versions from

    Returns:
        Tuple of (bashio_version, tempio_version, s6_overlay_version)
    """
    # Try to load from cache first
    cached_versions = load_cached_versions()
    if cached_versions:
        log_verbose("Using cached versions")
        return (
            cached_versions.get('BASHIO_VERSION'),
            cached_versions.get('TEMPIO_VERSION'),
            cached_versions.get('S6_OVERLAY_VERSION')
        )

    # Use master branch instead of main
    url = f"https://raw.githubusercontent.com/home-assistant/docker-base/master/ubuntu/build.yaml"
    raw_text = fetch_with_retry(url)

    if raw_text is None:
        log_warning("Using default versions due to network failure")
        return (
            DEFAULT_VERSIONS['BASHIO_VERSION'],
            DEFAULT_VERSIONS['TEMPIO_VERSION'],
            DEFAULT_VERSIONS['S6_OVERLAY_VERSION']
        )

    try:
        yaml_content = yaml.safe_load(raw_text)
        args = yaml_content.get('args', {})
        versions = {
            'BASHIO_VERSION': args.get('BASHIO_VERSION'),
            'TEMPIO_VERSION': args.get('TEMPIO_VERSION'),
            'S6_OVERLAY_VERSION': args.get('S6_OVERLAY_VERSION')
        }

        # Save to cache if we got valid versions
        if all(versions.values()):
            save_versions_to_cache(versions)
            return versions['BASHIO_VERSION'], versions['TEMPIO_VERSION'], versions['S6_OVERLAY_VERSION']

        log_warning("Missing version information in YAML file, using defaults")
        return (
            DEFAULT_VERSIONS['BASHIO_VERSION'],
            DEFAULT_VERSIONS['TEMPIO_VERSION'],
            DEFAULT_VERSIONS['S6_OVERLAY_VERSION']
        )

    except yaml.YAMLError as e:
        log_warning(f"Failed to parse YAML from {url}: {str(e)}")
        return (
            DEFAULT_VERSIONS['BASHIO_VERSION'],
            DEFAULT_VERSIONS['TEMPIO_VERSION'],
            DEFAULT_VERSIONS['S6_OVERLAY_VERSION']
        )


def create_package(version, default=False) -> list:
    pkg = package.copy()
    try:
        wanted_version = github_latest_tag('home-assistant/docker-base') if version == 'latest' else version
        bashio_version, tempio_version, s6_overlay_version = latest_deps_versions(wanted_version)
    except Exception as e:
        print(f"Failed to fetch the latest version of dependencies: {e}")
        bashio_version, tempio_version, s6_overlay_version = None, None, None

    if not all([bashio_version, tempio_version, s6_overlay_version]):
        log_warning("Failed to get dependency versions, using defaults")
        bashio_version = DEFAULT_VERSIONS['BASHIO_VERSION']
        tempio_version = DEFAULT_VERSIONS['TEMPIO_VERSION']
        s6_overlay_version = DEFAULT_VERSIONS['S6_OVERLAY_VERSION']

    pkg['name'] = f'homeassistant-base:{version}'
    pkg['build_args'] = {
        'BASHIO_VERSION': bashio_version,
        'TEMPIO_VERSION': tempio_version,
        'S6_OVERLAY_VERSION': s6_overlay_version,
    }

    if default:
        pkg['alias'] = 'homeassistant-base'

    return pkg

package = [
    create_package('latest', default=True),
]
