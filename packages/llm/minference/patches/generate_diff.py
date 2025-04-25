#!/usr/bin/env python3
import difflib
import os
import re

def modify_setup_py(original_content):
    lines = original_content.splitlines()
    plat_start = None
    plat_end = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def get_platform():"):
            plat_start = i
            for j in range(i+1, len(lines)):
                if lines[j] and not lines[j].startswith("    ") and not lines[j].strip().startswith("#"):
                    plat_end = j - 1
                    break
            else:
                plat_end = len(lines) - 1
            break

    if plat_start is not None and plat_end is not None:
        new_plat_block = [
            "def get_arch():",
            '    """',
            "    Returns the system aarch for the current system.",
            '    """',
            '    if sys.platform.startswith("linux"):',
            '        if platform.machine() == "x86_64":',
            '            return "x86_64"',
            '        if platform.machine() == "arm64" or platform.machine() == "aarch64":',
            '            return "aarch64"',
            '    elif sys.platform == "darwin":',
            '        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])',
            '        return f"macosx_{mac_version}_x86_64"',
            '    elif sys.platform == "win32":',
            '        return "win_amd64"',
            '    else:',
            '        raise ValueError("Unsupported platform: {}".format(sys.platform))',
            "",
            "def get_system() -> str:",
            '    """',
            "    Returns the system name as used in wheel filenames.",
            '    """',
            '    if platform.system() == "Windows":',
            '        return "win"',
            '    elif platform.system() == "Darwin":',
            '        mac_version = ".".join(platform.mac_ver()[0].split(".")[:1])',
            '        return f"macos_{mac_version}"',
            '    elif platform.system() == "Linux":',
            '        return "linux"',
            '    else:',
            '        raise ValueError("Unsupported system: {}".format(platform.system()))',
            "",
            "def get_platform() -> str:",
            '    """',
            "    Returns the platform name as used in wheel filenames.",
            '    """',
            '    return f"{get_system()}_{get_arch()}"',
        ]
        lines[plat_start:plat_end+1] = new_plat_block
    return "\n".join(lines) + "\n"

def generate_diff(original_file, output_diff):
    with open(original_file, "r") as f:
        original_content = f.read()
    modified_content = modify_setup_py(original_content)
    with open("setup_modified.py", "w") as f:
        f.write(modified_content)
    diff = difflib.unified_diff(
        original_content.splitlines(),
        modified_content.splitlines(),
        fromfile="a/setup.py",
        tofile="b/setup.py",
        lineterm=""
    )
    with open(output_diff, "w") as f:
        f.write("\n".join(diff) + "\n")

if __name__ == "__main__":
    generate_diff("/opt/minference/setup.py", "/tmp/minference/patch.diff")