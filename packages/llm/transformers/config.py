from jetson_containers import handle_json_request, github_latest_tag

def transformers(version, source=None, requires='>=34', default=False):
    pkg = package.copy()

    if version:
        pkg['name'] += f':{version}'

    pkg['build_args'] = {
        'TRANSFORMERS_PACKAGE': source,
        'TRANSFORMERS_VERSION': version
    }

    if requires:
        pkg['requires'] = requires

    if default:
        pkg['alias'] = f'transformers'

    return pkg


def transformers_pypi(version, **kwargs):
    releases = f'https://pypi.org/pypi/transformers/json'

    if version == 'latest':
        data = handle_json_request(releases)
        version = data['releases'].keys()[-1]

        print(f'releases: {data['releases'].keys()}')
        print(f'latest: {version}')

    return transformers(
        version,
        source=f'transformers',
        **kwargs
    )


def transformers_git(version, repo='huggingface/transformers', branch='main', **kwargs):
    version = github_latest_tag(repo) if version == 'latest' else version
    return transformers(
        version,
        source=f'git+https://github.com/{repo}',
        **kwargs
    )


# 11/3/23 - removing 'bitsandbytes' and 'auto_gptq' due to circular dependency and increased load times of
# anything using transformers if you want to use load_in_8bit/load_in_4bit or AutoGPTQ quantization 
# built-into transformers, use the 'bitsandbytes' or 'auto_gptq' containers directly instead of transformers container
package = [
    transformers_pypi('latest', default=True),                  # will always resolve to the latest pypi version
    transformers_pypi('4.35.0'),                                # For compatibility with AutoAWQ
    transformers_git('git'),
    transformers_git('nvgpt', repo='ertkonuk/transformers'),
]
