"""
Tests for the list-splitting helpers used by jetson_containers/build.py's
--skip-packages / --skip-tests / --test-only / --package-dirs argparse
handling.

build.py parses argv and runs build logic at import time, so we don't import
the module directly. Instead we import the small, pure helper functions it
exposes (split_package_list / split_path_list) directly from the file via
importlib, without triggering the module-level argparse/build code.
"""

import ast
import fnmatch
import re
from pathlib import Path

BUILD_PY = (
    Path(__file__).resolve().parents[1] / "jetson_containers" / "build.py"
)


def _load_helpers():
    """Extract just the helper function defs (and anything they need) from
    build.py's source and exec them in an isolated namespace, so we test the
    real implementation without running the rest of the script (which parses
    sys.argv and would try to build containers)."""
    source = BUILD_PY.read_text()
    tree = ast.parse(source, filename=str(BUILD_PY))

    wanted = {
        "split_package_list",
        "split_path_list",
        "unmatched_package_patterns",
    }
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            found[node.name] = ast.get_source_segment(source, node)

    missing = wanted - found.keys()
    assert not missing, (
        f"build.py must define {sorted(missing)} as importable module-level "
        "helper functions (list-splitting logic must be extracted, not "
        "inline at import time)."
    )

    namespace = {"re": re, "fnmatch": fnmatch}
    for name in sorted(found):
        exec(compile(found[name], str(BUILD_PY), "exec"), namespace)

    return (
        namespace["split_package_list"],
        namespace["split_path_list"],
        namespace["unmatched_package_patterns"],
    )


(
    split_package_list,
    split_path_list,
    unmatched_package_patterns,
) = _load_helpers()


# --- split_package_list: --skip-packages, --skip-tests, --test-only ---


def test_tagged_package_name_survives_split():
    # acceptance criterion: --skip-tests=cudastack:standard yields
    # ['cudastack:standard'], not ['cudastack', 'standard']
    assert split_package_list("cudastack:standard") == ["cudastack:standard"]


def test_comma_separated_list_parses_as_before():
    assert split_package_list("pytorch,tensorflow") == [
        "pytorch",
        "tensorflow",
    ]


def test_semicolon_separated_list_parses_as_before():
    assert split_package_list("pytorch;tensorflow") == [
        "pytorch",
        "tensorflow",
    ]


def test_mixed_comma_and_semicolon_list():
    assert split_package_list("pytorch,tensorflow;jupyterlab") == [
        "pytorch",
        "tensorflow",
        "jupyterlab",
    ]


def test_empty_string_yields_single_empty_element():
    # matches the historical re.split behavior on '' (used as the argparse
    # default), which downstream code treats as "no filter"
    assert split_package_list("") == [""]


def test_special_value_all_is_preserved():
    assert split_package_list("all") == ["all"]


def test_special_value_intermediate_is_preserved():
    assert split_package_list("intermediate") == ["intermediate"]


def test_special_values_alongside_tagged_names():
    assert split_package_list("intermediate,cudastack:standard") == [
        "intermediate",
        "cudastack:standard",
    ]


def test_multiple_tagged_names_comma_separated():
    assert split_package_list("cudastack:standard,vllm:builder") == [
        "cudastack:standard",
        "vllm:builder",
    ]


# --- split_path_list: used for --package-dirs only; ':' stays a separator ---


def test_package_dirs_still_splits_on_colon():
    assert split_path_list("/a/b:/c/d") == ["/a/b", "/c/d"]


def test_package_dirs_still_splits_on_comma_and_semicolon():
    assert split_path_list("/a/b,/c/d;/e/f") == ["/a/b", "/c/d", "/e/f"]


def test_package_dirs_empty_string_yields_single_empty_element():
    assert split_path_list("") == [""]


# --- documented --skip-tests / --skip-packages / --test-only examples parse
#     identically to how build.py used to parse them before this fix (no ':'
#     in play in them, so this is a stability guard, not a behavior-change
#     test). dusty-nv/jetson-containers has no CI SKIP_TESTS table (the nv
#     fork's version of this test reads .github/.../ci_config.py), so the
#     examples in docs/build.md are the reference values here. ---


def test_documented_filter_values_parse_identically_to_before():
    docs = Path(__file__).resolve().parents[1] / "docs" / "build.md"
    values = re.findall(
        r"--(?:skip-tests|skip-packages|test-only)=(\S+)", docs.read_text()
    )
    assert values, "expected at least one documented filter value to check"

    def old_split(s):  # build.py's behavior before this change
        return re.split(",|;|:", s)

    for value in values:
        assert split_package_list(value) == old_split(value), (
            f"split_package_list changed behavior for documented value "
            f"{value!r}"
        )


# --- unmatched_package_patterns: warn when a filter silently does nothing ---

_NAMES = [
    "numpy",
    "onnx",
    "pytorch:2.9.1",
    "cudastack:standard",
    "cudastack:full",
]


def test_old_colon_list_is_reported_as_unmatched():
    # 'numpy:onnx' used to mean two packages; now it's one pattern that
    # matches nothing
    patterns = split_package_list("numpy:onnx")
    assert unmatched_package_patterns(patterns, _NAMES) == ["numpy:onnx"]


def test_tagged_names_and_wildcards_match():
    patterns = split_package_list("cudastack:standard,pytorch:*,num*")
    assert unmatched_package_patterns(patterns, _NAMES) == []


def test_typo_is_reported_as_unmatched():
    patterns = split_package_list("numpy,pytorhc")
    assert unmatched_package_patterns(patterns, _NAMES) == ["pytorhc"]


def test_empty_and_special_values_are_not_reported():
    patterns = split_package_list("all,intermediate,")
    assert unmatched_package_patterns(
        patterns, _NAMES, special=("all", "intermediate")) == []
    assert unmatched_package_patterns(split_package_list(""), _NAMES) == []
