#!/usr/bin/env python3
import difflib
import os
import re
import argparse

def modify_CMakeLists(content):
    lines = content.splitlines()
    new_lines = []
    for line in lines:
        if re.match(r'^\s*set\s*\(\s*CUDA_SUPPORTED_ARCHS\s+["\'].*["\']\s*\)', line):
            new_lines.append('set(CUDA_SUPPORTED_ARCHS "8.7;10.1")')
            continue
        pattern = r'(cuda_archs_loose_intersection\(\s*\S+\s+)"[^"]+"\s+("[^"]+"\))'
        if re.search(pattern, line):
            newline = re.sub(pattern, r'\1"${CUDA_ARCHS}" \2', line)
            new_lines.append(newline)
            continue
        new_lines.append(line)
    return "\n".join(new_lines) + "\n"

def modify_vllm_flash_attn_cmake(content):
    lines = content.splitlines()
    new_lines = []
    patch_inserted = False
    for i, line in enumerate(lines):
        if not patch_inserted and re.match(r'^\s*if\s*\(DEFINED\s+ENV\{VLLM_FLASH_ATTN_SRC_DIR\}\)', line):
            new_lines.append('set(patch_vllm_flash_attn git apply /tmp/vllm/fa.diff)')
            patch_inserted = True
        new_lines.append(line)
        if re.search(r'^\s*BINARY_DIR\s+\$\{CMAKE_BINARY_DIR\}/vllm-flash-attn', line):
            new_lines.append('          PATCH_COMMAND ${patch_vllm_flash_attn}')
            new_lines.append('          UPDATE_DISCONNECTED 1')
    return "\n".join(new_lines) + "\n"

def modify_guided_decoding_init(content):
    lines = content.splitlines()
    new_lines = []
    skip = False
    for line in lines:
        if "xgrammar only has x86 wheels for linux" in line:
            skip = True
            continue
        if skip and "xgrammar doesn't support regex" in line:
            skip = False
            new_lines.append(line)
            continue
        if not skip:
            new_lines.append(line)
    return "\n".join(new_lines) + "\n"

def generate_diff_for_file(original_file, modify_function, base_dir, output_diff):
    try:
        with open(original_file, "r") as f:
            original_content = f.read()
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {original_file}")
        return
    modified_content = modify_function(original_content)
    relative_path = os.path.relpath(original_file, start=base_dir)
    diff = difflib.unified_diff(
        original_content.splitlines(),
        modified_content.splitlines(),
        fromfile="a/" + relative_path,
        tofile="b/" + relative_path,
        lineterm=""
    )
    with open(output_diff, "w") as f:
        f.write("\n".join(diff) + "\n")
    print(f"Diff generado para {relative_path} en {output_diff}")

def main():
    parser = argparse.ArgumentParser(description="Generate diffs for vLLM")
    parser.add_argument("--base-dir", default="/opt/vllm", help="Root Path Repository")
    args = parser.parse_args()
    base_dir = args.base_dir

    files_to_modify = {
        os.path.join(base_dir, "CMakeLists.txt"): modify_CMakeLists,
        os.path.join(base_dir, "cmake", "external_projects", "vllm_flash_attn.cmake"): modify_vllm_flash_attn_cmake,
    }
    diff_dir = "/tmp/vllm"
    os.makedirs(diff_dir, exist_ok=True)
    for filepath, mod_func in files_to_modify.items():
        diff_filename = os.path.basename(filepath) + ".diff"
        diff_file = os.path.join(diff_dir, diff_filename)
        generate_diff_for_file(filepath, mod_func, base_dir, diff_file)

if __name__ == "__main__":
    main()