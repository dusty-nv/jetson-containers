#!/usr/bin/env python3
import difflib

def modify_setup_py(original_content):
    lines = original_content.splitlines()
    
    # Replace get_platform() function
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

    # Replace add_cuda_gencodes() function
    cuda_func_start = None
    cuda_func_end = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def add_cuda_gencodes("):
            cuda_func_start = i
            # Find the end of the function by looking for the next function definition
            # or a line at module level (no indentation)
            func_indent = len(line) - len(line.lstrip())
            for j in range(i+1, len(lines)):
                current_line = lines[j]
                if not current_line.strip():
                    continue
                current_indent = len(current_line) - len(current_line.lstrip())
                # If we find a line at the same or less indentation that's not a comment
                # and it's not just whitespace, it's the end of the function
                if current_indent <= func_indent and current_line.strip() and not current_line.strip().startswith("#"):
                    cuda_func_end = j - 1
                    break
            else:
                cuda_func_end = len(lines) - 1
            break

    if cuda_func_start is not None and cuda_func_end is not None:
        new_cuda_func = [
            "def add_cuda_gencodes(cc_flag, archs, bare_metal_version):",
            '    """',
            "    Adds -gencode flags based on CUDA_ARCHITECTURES environment variable.",
            '    """',
            "    for CUDA_ARCH in os.environ.get('CUDA_ARCHITECTURES', '80;90;100;110;120').split(';'):",
            "        if CUDA_ARCH.strip():",
            "            cc_flag += [\"-gencode\", f\"arch=compute_{CUDA_ARCH.strip()},code=sm_{CUDA_ARCH.strip()}\"]",
            "    return cc_flag",
        ]
        lines[cuda_func_start:cuda_func_end+1] = new_cuda_func

    # Add type hints
    for i, line in enumerate(lines):
        if line.strip().startswith("def get_package_version():"):
            lines[i] = line.replace("def get_package_version():", "def get_package_version() -> str:")
        if line.strip().startswith("def get_wheel_url():"):
            lines[i] = line.replace("def get_wheel_url():", "def get_wheel_url() -> tuple[str, str]:")
        if line.strip().startswith("def run(self):") and i >= 2:
            if "class CachedWheelsCommand" in lines[i-2]:
                lines[i] = "    def run(self) -> None:"

    return "\n".join(lines) + "\n"

def generate_diff(original_file, output_diff):
    with open(original_file, "r", encoding="utf-8") as f:
        original_content = f.read()
    modified_content = modify_setup_py(original_content)
    with open("setup_modified.py", "w", encoding="utf-8") as f:
        f.write(modified_content)
    diff = difflib.unified_diff(
        original_content.splitlines(),
        modified_content.splitlines(),
        fromfile="a/setup.py",
        tofile="b/setup.py",
        lineterm=""
    )
    with open(output_diff, "w", encoding="utf-8") as f:
        f.write("\n".join(diff) + "\n")

if __name__ == "__main__":
    generate_diff("/opt/flash-attention/setup.py", "/tmp/flash-attention/patch.diff")