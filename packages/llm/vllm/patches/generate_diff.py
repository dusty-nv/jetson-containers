#!/usr/bin/env python3
import difflib
import os
import re
import argparse

def modify_CMakeLists(content: str) -> str:
    """
    - Replaces any `set(CUDA_SUPPORTED_ARCHS "...")` with `set(CUDA_ARCHS "<from-env-or-${CUDA_ARCHS}>")`
    - Rewrites every cuda_archs_loose_intersection(...) so that its first quoted argument
      becomes "${CUDA_ARCHS}", preserving the third argument.
      Handles both:
         * single-line calls
         * two-line calls (i.e. function name + var on one line, then "..." "..." ) on the next
    """
    # Prepare the replacement for any “set(CUDA_SUPPORTED_ARCHS "...")”
    cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', '"${CUDA_ARCHS}"')
    set_pattern = re.compile(
        r'^\s*set\s*\(\s*CUDA_SUPPORTED_ARCHS\s+["\'].*["\']\s*\)',
        flags=re.MULTILINE
    )
    # First, replace any “set(CUDA_SUPPORTED_ARCHS "…")” with “set(CUDA_ARCHS "<cuda_arch_list>")”
    content = set_pattern.sub(f'set(CUDA_ARCHS {cuda_arch_list})', content)

    # Now we want to catch every cuda_archs_loose_intersection, whether it’s on one line or two.
    lines = content.splitlines(keepends=True)
    new_lines = []

    # 1) Pattern to detect the start of a two-line invocation:
    #    e.g.  “    cuda_archs_loose_intersection(SCALED_MM_2X_ARCHS\n”
    start_two_line_re = re.compile(
        r'^\s*cuda_archs_loose_intersection\s*\(\s*(\w+)\s*$'
    )
    # 2) Pattern for the second line of a two-line invocation:
    #    e.g.  “      "7.5;8.0;8.9\+PTX"  "${CUDA_ARCHS}")\n”
    args_two_line_re = re.compile(
        r'^\s*"([^"]+)"\s*"([^"]+)"\s*\)\s*$'
    )

    # 3) Pattern to catch ANY single-line invocation:
    #    e.g.  cuda_archs_loose_intersection(CUDA_ARCHS "7.0;7.2" "8.0;8.6")
    single_line_re = re.compile(
        r'^\s*(cuda_archs_loose_intersection\s*\(\s*\w+)\s*"([^"]+)"\s*"([^"]+)"\s*(\))'
    )

    i = 0
    while i < len(lines):
        line = lines[i]

        # ——— Handle two-line invocation ———
        m_start = start_two_line_re.match(line)
        if m_start and (i + 1) < len(lines):
            varname = m_start.group(1)
            next_line = lines[i + 1]
            m_args = args_two_line_re.match(next_line)
            if m_args:
                old_first = m_args.group(1)   # e.g.  "7.5;8.0;8.9+PTX"
                old_second = m_args.group(2)  # e.g.  "${CUDA_ARCHS}"
                indent = re.match(r'^(\s*)', line).group(1)

                # Build a single-line replacement:
                #   <indent>cuda_archs_loose_intersection(VAR "${CUDA_ARCHS}" "old_second")\n
                new_lines.append(
                    f'{indent}cuda_archs_loose_intersection({varname} "{cuda_arch_list}" "{old_second}")\n'
                )
                i += 2
                continue

        # ——— Handle single-line invocation ———
        m_single = single_line_re.match(line)
        if m_single:
            # m_single.groups() == ( "cuda_archs_loose_intersection(VAR", old_first, old_second, ")" )
            func_and_var = m_single.group(1)      # e.g.  "cuda_archs_loose_intersection(CUDA_ARCHS"
            old_second = m_single.group(3)        # e.g.  "8.0;8.6"  or "${CUDA_ARCHS}"
            trailing_paren = m_single.group(4)    # just “)”
            indent = re.match(r'^(\s*)', line).group(1)

            # Replace with: cuda_archs_loose_intersection(VAR "${CUDA_ARCHS}" "old_second")
            new_lines.append(
                f'{indent}{func_and_var} "{cuda_arch_list}" "{old_second}"{trailing_paren}\n'
            )
            i += 1
            continue

        # ——— Otherwise, copy the line verbatim ———
        new_lines.append(line)
        i += 1

    return ''.join(new_lines)

def modify_vllm_flash_attn_cmake(content):
    lines = content.splitlines()
    new_lines = []

    patch_inserted = False
    env_block_open = False

    for line in lines:
        if not patch_inserted and re.match(r'^\s*if\s*\(DEFINED\s+ENV\{VLLM_FLASH_ATTN_SRC_DIR\}\)', line):
            env_block_open = True

        new_lines.append(line)

        if env_block_open and not patch_inserted and re.match(r'^\s*endif\s*\(\s*\)\s*$', line):
            fa_patch_name = 'fa.diff'
            cuda_arch = os.environ.get('TORCH_CUDA_ARCH_LIST')
            # Specific patch for sm_87 by vLLM version
            if cuda_arch == '8.7':
                vllm_version = os.environ.get('VLLM_VERSION')
                fa_patch_name = 'sm_87-' + vllm_version + '-' + fa_patch_name
            new_lines.append('set(patch_vllm_flash_attn git apply /tmp/vllm/' + fa_patch_name + ')')
            patch_inserted = True
            env_block_open = False
        if re.search(r'^\s*BINARY_DIR\s+\$\{CMAKE_BINARY_DIR\}/vllm-flash-attn', line):
            new_lines.append('          PATCH_COMMAND ${patch_vllm_flash_attn}')
            new_lines.append('          UPDATE_DISCONNECTED 1')

    return "\n".join(new_lines) + "\n"

def modify_guided_decoding_init(content):
    # Removes block between xgrammar comments
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
        print(f"Error: File not found {original_file}")
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
    print(f"Diff generated for {relative_path} on {output_diff}")

def main():
    parser = argparse.ArgumentParser(description="Generate diffs for vLLM")
    parser.add_argument("--base-dir", default="/opt/vllm", help="Root Path Repository")
    args = parser.parse_args()
    base_dir = args.base_dir

    files_to_modify = {
        os.path.join(base_dir, "CMakeLists.txt"): modify_CMakeLists,
        os.path.join(base_dir, "cmake", "external_projects", "vllm_flash_attn.cmake"): modify_vllm_flash_attn_cmake,
        # You can add more files as needed:
        # os.path.join(base_dir, ...): modify_guided_decoding_init,
    }
    diff_dir = "/tmp/vllm"
    os.makedirs(diff_dir, exist_ok=True)
    for filepath, mod_func in files_to_modify.items():
        diff_filename = os.path.basename(filepath) + ".diff"
        diff_file = os.path.join(diff_dir, diff_filename)
        generate_diff_for_file(filepath, mod_func, base_dir, diff_file)

if __name__ == "__main__":
    main()
