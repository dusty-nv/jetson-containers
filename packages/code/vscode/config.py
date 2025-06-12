from jetson_containers import SYSTEM_ARM, package_depends


def vscode(tag, depends=[]):
    """
    Define containers that have a portable version of VSCode installed, in addition
    to other common depenencies used during development like CUDA and HF Transformers.
    """
    pkg = package.copy()
    pkg['name'] = tag

    # https://code.visualstudio.com/docs/editor/portable
    pkg['build_args'] = {
        'VSCODE_URL': 'https://code.visualstudio.com/sha/download?build=stable&os=linux-' + \
            'arm64' if SYSTEM_ARM else 'x64'
    }

    if depends:
        package_depends(pkg, depends)

    return pkg   


package = [
    vscode('vscode'),
    vscode('vscode:cuda', depends=['cuda', 'cudnn', 'tensorrt']),
    vscode('vscode:torch', depends=['pytorch', 'torchvision', 'torchaudio', 'torchao', 'torch2trt']),
    vscode('vscode:transformers', depends=['transformers', 'flash-attention', 'bitsandbytes', 'triton']),
]