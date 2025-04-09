from jetson_containers import L4T_VERSION, github_latest_tag, log_warning
from jetson_containers.pypi_utils import get_latest_version
from packaging.version import parse

def transformers_pypi(version, **kwargs):
    if version == 'latest':
        version = get_latest_version('transformers')
        if not version:
            log_warning("Failed to fetch latest transformers version from PyPI")
            return None

    return _transformers(
        version,
        source=f'transformers=={version}',
        **kwargs
    )

# Rest of the file remains unchanged...