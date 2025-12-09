import os

from jetson_containers import JETPACK_VERSION


def get_latest_release_tag(repo_path):
    """Get the latest release tag from a GitHub repository.

    Args:
        repo_path: GitHub repo path like user/repo

    Returns:
        str: Latest release tag name
    """

    # Call GitHub API
    api_url = f"https://api.github.com/repos/{repo_path}/releases/latest"
    import requests
    response = requests.get(api_url)

    if response.status_code == 200:
        return response.json()['tag_name']
    else:
        raise Exception(
            f"Failed to fetch release from {api_url} : {response.status_code} - {response.text}")


def open_webui(version, repo=None, requires=None, default=False):
    if repo and version == 'stable':
        version = get_latest_release_tag(repo)
        # strip the starting v if exists
        if version.startswith('v'):
            version = version[1:]

    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'open-webui:{version}'
    use_ollama = os.environ.get('USE_OLLAMA', 'false')
    if use_ollama.lower() == 'true' or use_ollama == '1' or use_ollama.lower() == 'on':
        use_ollama = 'true'
    else:
        use_ollama = 'false'

    pkg['build_args'] = {
        'OPEN_WEBUI_VERSION': version,
        'USE_CUDA': 'true',
        'BUILDPLATFORM': 'linux/arm64',
        # Passing 'USE_OLLAMA=true' to install ollama client in the same image as webui.
        # Usage: USE_OLLAMA=true jetson-containers build open-webui
        'USE_OLLAMA': use_ollama,
        'JETSON_JETPACK': JETPACK_VERSION.major,
    }

    builder = pkg.copy()

    builder['name'] = f'open-webui:{version}-builder'

    if default:
        pkg['alias'] = 'open-webui'
        builder['alias'] = 'open-webui:builder'

    return pkg, builder


package = [
    open_webui('stable', repo='open-webui/open-webui',default=True),
]
