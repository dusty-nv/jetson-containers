def create_package(version=None, default=False) -> list:
    pkg = package.copy()

    if version is not None:
        pkg['name'] = f'python:{version}'
        pkg['build_args'] = {
            'DEADSNAKES_PYTHON_VERSION': version,
        }

    if default:
        pkg['alias'] = 'python'

    return pkg


package = [
    # default core python
    create_package(default=True),
    # deadsnakes python
    create_package(version='3.11'),
    create_package(version='3.12'),
]
