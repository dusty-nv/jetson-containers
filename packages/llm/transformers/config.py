from jetson_containers import handle_json_request, github_latest_tag
from packaging.version import parse


def _transformers(version, codename=None, source=None, requires='>=34', default=False):
    pkg = package.copy()

    pkg['name'] = f'{package["name"]}:{codename or version}'

    pkg['build_args'] = {
        'TRANSFORMERS_PACKAGE': source,
        'TRANSFORMERS_VERSION': version
    }

    if requires:
        pkg['requires'] = requires

    if default:
        pkg['alias'] = package['name']

    return pkg


def transformers_pypi(version, **kwargs):
    if version == 'latest':
        data = handle_json_request('https://pypi.org/pypi/transformers/json')
        # Sort the version keys using `parse` for proper semantic version sorting
        sorted_versions = sorted(data['releases'].keys(), key=parse)
        version = sorted_versions[-1]

    return _transformers(
        version,
        source=f'transformers=={version}',
        **kwargs
    )


def transformers_git(version, repo='huggingface/transformers', branch=None, **kwargs):
    version = github_latest_tag(repo) if version == 'latest' else version

    if version.startswith('v'):
        version = version[1:]

    if branch is None:
        branch = f'v{version}'

    return _transformers(
        version,
        source=f'git+https://github.com/{repo}@{branch}',
        **kwargs
    )

# 11/3/23 - removing 'bitsandbytes' and 'auto_gptq' due to circular dependency and increased load times of
# anything using transformers if you want to use load_in_8bit/load_in_4bit or AutoGPTQ quantization 
# built-into transformers, use the 'bitsandbytes' or 'auto_gptq' containers directly instead of transformers container
package = [
    transformers_pypi('latest', default=True),                                  # will always resolve to the latest pypi version
    transformers_git('latest', codename='git'),                                                 # will always resolve to the latest git version from huggingface/transformers
    transformers_git('latest', codename='nvgpt', repo='ertkonuk/transformers'),     # will always resolve to the latest git version from ertkonuk/transformers
]
