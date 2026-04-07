#!/usr/bin/env python3
"""
Dynamically generate Jetson-compatible patches for vLLM.

Reads the cloned vLLM source, applies arch/version transformations, and outputs:
  - /tmp/vllm/patch.diff   (combined diff for vLLM source modifications)
  - /tmp/vllm/fa.diff      (diff for vendored flash-attention CMakeLists.txt)

Environment variables used:
  TORCH_CUDA_ARCH_LIST  – target CUDA SM (e.g. "8.7"), defaults to "${CUDA_ARCHS}"
  VLLM_VERSION          – vLLM version being built
"""
import difflib
import os
import re
import argparse
import subprocess
import urllib.request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_torch_version():
    """Return the installed PyTorch version (without +cu suffix)."""
    try:
        out = subprocess.run(
            ["python3", "-c",
             "import torch; print(torch.__version__.split('+')[0])"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _cuda_arch_cmake_value():
    """Raw value (no CMake quotes) for CUDA arch replacement."""
    return os.environ.get("TORCH_CUDA_ARCH_LIST", "${CUDA_ARCHS}")


def _unified_diff(original, modified, rel_path):
    """Return a unified-diff string (empty string when files are identical)."""
    diff_lines = list(difflib.unified_diff(
        original.splitlines(),
        modified.splitlines(),
        fromfile="a/" + rel_path,
        tofile="b/" + rel_path,
        lineterm="",
    ))
    if not diff_lines:
        return ""
    return "\n".join(diff_lines)


def _read(path):
    try:
        with open(path, "r") as fh:
            return fh.read()
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Modification functions – each takes file content and returns modified content
# ---------------------------------------------------------------------------

def modify_cmake_archs(content: str) -> str:
    """
    Patch a CMakeLists.txt so CUDA arch selection is driven by the env.

    * Collapses any ``set(CUDA_SUPPORTED_ARCHS ...)`` block (including
      multi-branch if/elseif/else/endif conditionals) into a single
      ``set(CUDA_ARCHS <value>)``.
    * Rewrites every ``cuda_archs_loose_intersection(VAR "..." "...")``
      so the first quoted argument becomes the env-driven value.
    * Updates ``TORCH_SUPPORTED_VERSION_CUDA`` to match the installed torch.
    """
    cuda_val = _cuda_arch_cmake_value()

    # ── 1. Collapse CUDA_SUPPORTED_ARCHS blocks ──────────────────────────
    lines = content.splitlines(keepends=True)
    out = []
    i = 0
    replaced_archs = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Pattern A – if/elseif block that *contains* set(CUDA_SUPPORTED_ARCHS)
        if (not replaced_archs
                and re.match(r"if\s*\(", stripped)
                and "CUDA_SUPPORTED_ARCHS" not in stripped):
            # Peek ahead for a set(CUDA_SUPPORTED_ARCHS ...) inside this block
            j = i + 1
            found_set = False
            endif_idx = None
            depth = 1
            while j < len(lines):
                s = lines[j].strip()
                if re.match(r"if\s*\(", s):
                    depth += 1
                if s.startswith("endif"):
                    depth -= 1
                    if depth == 0:
                        endif_idx = j
                        break
                if re.search(r"set\s*\(\s*CUDA_SUPPORTED_ARCHS\b", s):
                    found_set = True
                j += 1

            if found_set and endif_idx is not None:
                indent = re.match(r"^(\s*)", lines[i]).group(1)
                out.append(f'{indent}set(CUDA_ARCHS "{cuda_val}")\n')
                i = endif_idx + 1
                replaced_archs = True
                continue

        # Pattern B – standalone set(CUDA_SUPPORTED_ARCHS "...")
        if not replaced_archs and re.match(
                r"\s*set\s*\(\s*CUDA_SUPPORTED_ARCHS\s+", stripped):
            indent = re.match(r"^(\s*)", line).group(1)
            out.append(f'{indent}set(CUDA_ARCHS "{cuda_val}")\n')
            i += 1
            # Consume a trailing conditional-append block if present
            while i < len(lines):
                s = lines[i].strip()
                if (s.startswith("if(") or s.startswith("elseif(")
                        or re.match(r"list\s*\(\s*APPEND\s+CUDA_SUPPORTED_ARCHS", s)
                        or s.startswith("endif(") or s == ""):
                    is_endif = s.startswith("endif(")
                    i += 1
                    if is_endif:
                        break
                else:
                    break
            replaced_archs = True
            continue

        out.append(line)
        i += 1

    # ── 2. Rewrite cuda_archs_loose_intersection first-args ──────────────
    # Only replace the first argument when the target arch is >= the minimum
    # arch in the original list.  When the target is below the minimum (e.g.
    # target 8.7 but the line requires "9.0a"), leave the line untouched so
    # the intersection with CUDA_ARCHS is naturally empty and the feature is
    # skipped instead of attempting to compile unsupported kernels.
    lines = out
    out = []
    i = 0

    two_line_start = re.compile(
        r"^\s*cuda_archs_loose_intersection\s*\(\s*(\w+)\s*$")
    two_line_args = re.compile(r'^\s*"([^"]+)"\s*"([^"]+)"\s*\)\s*$')
    single_line = re.compile(
        r'^(\s*)(cuda_archs_loose_intersection\s*\(\s*\w+)\s*"([^"]+)"\s*"([^"]+)"\s*(\))')

    def _parse_arch_version(spec):
        """Extract numeric (major, minor) from an arch spec like '9.0a'."""
        m = re.match(r"(\d+)\.(\d+)", spec.strip())
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return None

    # Parse target arch; if it's a CMake variable we can't compare, so
    # fall back to replacing everything (original behaviour).
    target_version = _parse_arch_version(cuda_val)

    def _should_replace(first_arg):
        """True when any arch in first_arg has the same or lower major version
        as the target.  e.g. target 8.7 replaces "8.9;12.0;12.1" (because of
        8.9) but NOT "9.0a" (all archs have a higher major version)."""
        if target_version is None:
            return True  # can't compare, replace unconditionally
        target_major = target_version[0]
        for spec in first_arg.split(";"):
            v = _parse_arch_version(spec)
            if v is not None and v[0] <= target_major:
                return True
        return False

    while i < len(lines):
        line = lines[i]

        # Two-line form
        m = two_line_start.match(line)
        if m and (i + 1) < len(lines):
            var = m.group(1)
            m2 = two_line_args.match(lines[i + 1])
            if m2:
                if _should_replace(m2.group(1)):
                    indent = re.match(r"^(\s*)", line).group(1)
                    out.append(
                        f'{indent}cuda_archs_loose_intersection({var}'
                        f' "{cuda_val}" "{m2.group(2)}")\n')
                    i += 2
                    continue

        # Single-line form
        ms = single_line.match(line)
        if ms:
            indent, fn_var, first_arg, second, paren = ms.groups()
            if _should_replace(first_arg):
                out.append(
                    f'{indent}{fn_var} "{cuda_val}" "{second}"{paren}\n')
                i += 1
                continue

        out.append(line)
        i += 1

    content = "".join(out)

    # ── 3. Update TORCH_SUPPORTED_VERSION_CUDA ───────────────────────────
    torch_ver = _get_torch_version()
    if torch_ver:
        content = re.sub(
            r'set\(TORCH_SUPPORTED_VERSION_CUDA\s+"[^"]+"\)',
            f'set(TORCH_SUPPORTED_VERSION_CUDA "{torch_ver}")',
            content,
        )

    return content


def modify_vllm_flash_attn_cmake(content: str) -> str:
    """
    Insert ``PATCH_COMMAND git apply /tmp/vllm/fa.diff`` into both
    FetchContent_Declare blocks for vllm-flash-attn.
    """
    lines = content.splitlines()
    out = []

    patch_var_inserted = False
    env_block_open = False

    for line in lines:
        if (not patch_var_inserted
                and re.match(r"\s*if\s*\(DEFINED\s+ENV\{VLLM_FLASH_ATTN_SRC_DIR\}", line)):
            env_block_open = True

        out.append(line)

        # After the endif() of the env-variable block, inject the variable
        if env_block_open and not patch_var_inserted and re.match(r"\s*endif\s*\(\s*\)\s*$", line):
            out.append("set(patch_vllm_flash_attn git apply /tmp/vllm/fa.diff)")
            patch_var_inserted = True
            env_block_open = False

        # After every BINARY_DIR line inside FetchContent_Declare, add PATCH
        if re.search(r"^\s*BINARY_DIR\s+\$\{CMAKE_BINARY_DIR\}/vllm-flash-attn", line):
            out.append("          PATCH_COMMAND ${patch_vllm_flash_attn}")
            out.append("          UPDATE_DISCONNECTED 1")

    return "\n".join(out) + "\n"


def modify_guided_decoding_init(content: str) -> str:
    """Remove the xgrammar x86-only CPU architecture check (if still present)."""
    if "xgrammar only has x86 wheels for linux" not in content:
        return content

    lines = content.splitlines()
    out = []
    skip = False
    for line in lines:
        if "xgrammar only has x86 wheels for linux" in line:
            skip = True
            continue
        if skip and "xgrammar doesn't support regex" in line:
            skip = False
            out.append(line)
            continue
        if not skip:
            out.append(line)
    return "\n".join(out) + "\n"


def modify_vllm_utils_init(content: str) -> str:
    """
    Add shared system-memory handling for Jetson Thor (SM 11.0) and
    Spark (SM 12.1) which share system and device memory.
    """
    if "_get_device_sm" in content:
        return content
    if "class MemorySnapshot" not in content:
        return content

    lines = content.splitlines()
    out = []

    helper_injected = False
    for i, line in enumerate(lines):
        # Inject the helper before the MemorySnapshot class (and its decorators)
        if (not helper_injected
                and re.match(r"^(\s*)class MemorySnapshot", line)):
            # Walk back over any decorators already appended
            insert_pos = len(out)
            while insert_pos > 0 and out[insert_pos - 1].strip().startswith("@"):
                insert_pos -= 1
            helper_block = [
                "",
                "def _get_device_sm():",
                "    if torch.cuda.is_available():",
                "        major, minor = torch.cuda.get_device_capability()",
                "        return major * 10 + minor",
                "    return 0",
                "",
                "",
            ]
            out[insert_pos:insert_pos] = helper_block
            helper_injected = True

        out.append(line)

        # After the torch.cuda.mem_get_info() line, add the override
        if "self.free_memory, self.total_memory = torch.cuda.mem_get_info()" in line:
            indent = re.match(r"^(\s*)", line).group(1)
            out += [
                f"{indent}shared_sysmem_device_mem_sms = (110, 121)  # Thor, Spark",
                f"{indent}if _get_device_sm() in shared_sysmem_device_mem_sms:",
                f"{indent}    self.free_memory = psutil.virtual_memory().available",
            ]

    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Flash-attention diff generation
# ---------------------------------------------------------------------------

def _extract_fa_git_tag(cmake_path):
    """Parse the flash-attention GIT_TAG (40-hex commit) from vllm_flash_attn.cmake."""
    content = _read(cmake_path)
    if not content:
        return None, None
    repo_m = re.search(r"GIT_REPOSITORY\s+(https://github\.com/\S+)", content)
    tag_m = re.search(r"GIT_TAG\s+([a-f0-9]{40})", content)
    return (
        repo_m.group(1) if repo_m else None,
        tag_m.group(1) if tag_m else None,
    )


def _fetch_fa_cmake(repo_url, git_tag):
    """Download flash-attention CMakeLists.txt from GitHub."""
    raw = repo_url.replace("github.com", "raw.githubusercontent.com")
    if raw.endswith(".git"):
        raw = raw[:-4]
    url = f"{raw}/{git_tag}/CMakeLists.txt"
    print(f"Fetching flash-attention CMakeLists.txt from {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "jetson-containers"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except Exception as exc:
        print(f"  Warning: fetch failed ({exc}), trying git archive…")

    # Fallback: shallow clone
    try:
        tmp = "/tmp/_fa_src"
        subprocess.run(
            ["git", "clone", "--depth=1", "--filter=blob:none",
             "--no-checkout", repo_url, tmp],
            check=True, capture_output=True, timeout=60,
        )
        subprocess.run(
            ["git", "-C", tmp, "fetch", "origin", git_tag, "--depth=1"],
            check=True, capture_output=True, timeout=60,
        )
        subprocess.run(
            ["git", "-C", tmp, "checkout", git_tag, "--", "CMakeLists.txt"],
            check=True, capture_output=True, timeout=30,
        )
        return _read(os.path.join(tmp, "CMakeLists.txt"))
    except Exception as exc:
        print(f"  Warning: git fallback also failed ({exc})")
    return None


def generate_fa_diff(base_dir, diff_dir):
    """Produce ``fa.diff`` for the vendored flash-attention CMakeLists.txt."""
    cmake_path = os.path.join(
        base_dir, "cmake", "external_projects", "vllm_flash_attn.cmake")
    repo_url, git_tag = _extract_fa_git_tag(cmake_path)

    if not repo_url or not git_tag:
        print("Warning: could not determine flash-attention repo/tag – skipping fa.diff")
        return False

    original = _fetch_fa_cmake(repo_url, git_tag)
    if not original:
        print("Warning: could not fetch flash-attention source – skipping fa.diff")
        return False

    modified = modify_cmake_archs(original)
    diff_text = _unified_diff(original, modified, "CMakeLists.txt")
    if not diff_text:
        print("flash-attention CMakeLists.txt: no changes needed")
        return True

    out_path = os.path.join(diff_dir, "fa.diff")
    with open(out_path, "w") as fh:
        fh.write(diff_text + "\n")
    print(f"Generated {out_path}")
    return True


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate dynamic Jetson patches for vLLM")
    parser.add_argument(
        "--base-dir", default="/opt/vllm",
        help="Root of the cloned vLLM repository")
    parser.add_argument(
        "--output-dir", default="/tmp/vllm",
        help="Directory for generated .diff files")
    args = parser.parse_args()

    base = args.base_dir
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # ── vLLM source patches ──────────────────────────────────────────────
    targets = [
        ("CMakeLists.txt", modify_cmake_archs),
        (os.path.join("cmake", "external_projects", "vllm_flash_attn.cmake"),
         modify_vllm_flash_attn_cmake),
        (os.path.join("vllm", "model_executor", "guided_decoding", "__init__.py"),
         modify_guided_decoding_init),
        (os.path.join("vllm", "utils", "__init__.py"),
         modify_vllm_utils_init),
    ]

    diffs = []
    for rel_path, mod_fn in targets:
        full = os.path.join(base, rel_path)
        original = _read(full)
        if original is None:
            print(f"Skipping (not found): {rel_path}")
            continue
        modified = mod_fn(original)
        d = _unified_diff(original, modified, rel_path)
        if d:
            diffs.append(d)
            print(f"Patch generated for {rel_path}")
        else:
            print(f"No changes for {rel_path}")

    patch_path = os.path.join(out, "patch.diff")
    with open(patch_path, "w") as fh:
        fh.write("\n".join(diffs) + "\n")
    print(f"Combined patch → {patch_path}")

    # ── Flash-attention CMakeLists.txt patch ─────────────────────────────
    generate_fa_diff(base, out)


if __name__ == "__main__":
    main()
