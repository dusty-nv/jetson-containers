import os


def open_webui(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'open-webui:{version}'

    pkg['build_args'] = {
        'OPEN_WEBUI_VERSION': version,
        'USE_CUDA': 'true',
        'BUILDPLATFORM': 'linux/arm64',
        # Passing 'USE_OLLAMA=true' to install ollama client in the same image as webui.
        # Usage: USE_OLLAMA=true jetson-containers build open-webui
        'USE_OLLAMA': os.environ.get('USE_OLLAMA', 'false')
    }

    builder = pkg.copy()

    builder['name'] = f'open-webui:{version}-builder'

    if default:
        pkg['alias'] = 'open-webui'
        builder['alias'] = 'open-webui:builder'

    return pkg, builder


package = [
    open_webui('0.6.26', default=True),
]
