from jetson_containers import L4T_VERSION, handle_json_request, github_latest_tag
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

    if version == 'latest':
        version = github_latest_tag(repo) 
        if not version:
            print(f'-- Failed to get latest Transformers github tag for {repo}')
            return
        
    if version.startswith('v'):
        version = version[1:]

    if branch is None:
        branch = version
        if branch[0].isnumeric():
            branch = 'v' + branch

    return _transformers(
        version,
        source=f'git+https://github.com/{repo}@{branch}',
        **kwargs
    )

# 11/3/23 - removing 'bitsandbytes' and 'auto_gptq' due to circular dependency and increased load times of
# anything using transformers if you want to use load_in_8bit/load_in_4bit or AutoGPTQ quantization 
# built-into transformers, use the 'bitsandbytes' or 'auto_gptq' containers directly instead of transformers container
package = [
    transformers_pypi('latest', default=(L4T_VERSION.major >= 36), requires='>=36'), # will always resolve to the latest pypi version
    transformers_pypi('4.46.3', default=(L4T_VERSION.major < 36), requires='<36'),   # 4.46.3 is the last version that supports Python 3.8
    transformers_git('latest', codename='git'),                                      # will always resolve to the latest git version from huggingface/transformers
    transformers_git('main', requires='>=36'),                                       # will always resolve to the latest commit in main branch from github
]
