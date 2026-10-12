"""
Tests for jetson_containers/packages.py's cuda_libs package-key support
(task t2 of the cudastack-chain-aware plan; covers c2, h2).

packages.py uses relative imports (from .l4t_version import ..., etc.), so
it can't be loaded standalone by file path the way cudastack_chain.py is in
test_cudastack_chain.py. Instead we register a placeholder module in
sys.modules under the name "jetson_containers" (pointing __path__ at the
real package directory) and then import the "jetson_containers.packages"
submodule normally. Because the placeholder is already in sys.modules,
Python's import machinery never executes the real jetson_containers/
__init__.py, so utils.check_dependencies() (which can try a pip install /
hit the network) never runs. jetson_containers/packages.py itself has no
third-party import side effects beyond stdlib + yaml + packaging, both of
which are ordinary runtime deps here.
"""

import importlib
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_DIR = _REPO_ROOT / "jetson_containers"


def _stub_missing_display_deps():
    """
    jetson_containers/logging.py does `import tabulate` and `import
    termcolor` purely for console pretty-printing; jetson_containers/
    packages.py pulls that in transitively via `from .logging import
    log_debug, log_warning, log_error`. Neither is a runtime dependency of
    this repo's pyproject (`dependencies = []`, see CLAUDE.md), and neither
    is used by anything packages.py actually calls, so if they aren't
    installed, register tiny stand-ins instead of skipping/requiring them.
    Leaves a real installed tabulate/termcolor alone.
    """
    if "tabulate" not in sys.modules:
        try:
            import tabulate  # noqa: F401
        except ImportError:
            stub = types.ModuleType("tabulate")
            stub.tabulate = lambda *a, **k: ""
            sys.modules["tabulate"] = stub

    if "termcolor" not in sys.modules:
        try:
            import termcolor  # noqa: F401
        except ImportError:
            stub = types.ModuleType("termcolor")
            stub.colored = lambda text, *a, **k: text
            stub.cprint = lambda text, *a, **k: print(text)
            sys.modules["termcolor"] = stub


def _load_packages_module():
    """
    Import jetson_containers.packages without running jetson_containers/
    __init__.py. Reuses an already-registered "jetson_containers" package
    stand-in across calls within the same test session (sys.modules is
    process-global), but always re-imports the "packages" submodule so each
    test gets a subsequent import cached the same way the real package
    would behave.
    """
    _stub_missing_display_deps()

    if "jetson_containers" not in sys.modules:
        fake_pkg = types.ModuleType("jetson_containers")
        fake_pkg.__path__ = [str(_PKG_DIR)]
        sys.modules["jetson_containers"] = fake_pkg

    return importlib.import_module("jetson_containers.packages")


packages = _load_packages_module()


# ---------------------------------------------------------------------------
# _PACKAGE_KEYS / validate_dict
# ---------------------------------------------------------------------------


def test_cuda_libs_is_a_known_package_key():
    assert "cuda_libs" in packages._PACKAGE_KEYS


def test_validate_dict_accepts_cuda_libs_key():
    assert (
        packages.validate_dict({"name": "foo", "cuda_libs": ["cudnn"]}) is True
    )


def test_validate_dict_still_rejects_unknown_keys():
    assert (
        packages.validate_dict({"name": "foo", "totally_bogus_key": 1})
        is False
    )


# ---------------------------------------------------------------------------
# validate_lists: bare string -> one-element list
# ---------------------------------------------------------------------------


def test_validate_lists_normalizes_bare_string_cuda_libs():
    pkg = {"name": "foo", "cuda_libs": "cudnn"}
    result = packages.validate_lists(pkg)
    assert result["cuda_libs"] == ["cudnn"]


def test_validate_lists_leaves_cuda_libs_list_untouched():
    pkg = {"name": "foo", "cuda_libs": ["cudnn", "nccl"]}
    result = packages.validate_lists(pkg)
    assert result["cuda_libs"] == ["cudnn", "nccl"]


def test_validate_lists_without_cuda_libs_is_unaffected():
    pkg = {"name": "foo", "alias": "bar"}
    result = packages.validate_lists(pkg)
    assert result["alias"] == ["bar"]
    assert "cuda_libs" not in result


# ---------------------------------------------------------------------------
# Unknown cuda_libs entries: loud (logged) failure that disables just that
# package -- not a raise (which aborts the whole scan on python 3.8 and kills
# a scan thread on newer pythons), and not a silent drop
# ---------------------------------------------------------------------------


def test_validate_lists_disables_package_on_unknown_cuda_lib(capsys):
    pkg = {"name": "my-package", "cuda_libs": ["cudnn", "not_a_real_lib"]}

    packages.validate_lists(pkg)

    assert pkg["disabled"] is True
    out = capsys.readouterr()
    message = out.out + out.err
    assert "my-package" in message
    assert "not_a_real_lib" in message


def test_validate_lists_disables_package_on_unknown_bare_string_cuda_lib(
    capsys,
):
    pkg = {"name": "my-package", "cuda_libs": "not_a_real_lib"}

    packages.validate_lists(pkg)

    assert pkg["disabled"] is True
    out = capsys.readouterr()
    assert "not_a_real_lib" in out.out + out.err


def test_validate_lists_known_cuda_libs_do_not_disable():
    pkg = {"name": "my-package", "cuda_libs": ["cudnn", "nccl"]}

    packages.validate_lists(pkg)

    assert "disabled" not in pkg


def test_validate_lists_bare_cuda_libs_key_means_declared_empty():
    """
    A bare `# cuda_libs:` header loads as None. compute_set() treats None as
    "declared, needs nothing", so validate_lists must accept it (normalized to
    []) rather than crash iterating None.
    """
    pkg = {"name": "my-package", "cuda_libs": None}

    packages.validate_lists(pkg)

    assert pkg["cuda_libs"] == []
    assert "disabled" not in pkg


def test_validate_package_drops_package_with_unknown_cuda_lib(monkeypatch):
    """
    A bad cuda_libs value set by a config.py (i.e. only seen by the final
    validate_package() pass) must still remove the package, so validate_lists
    has to run before the disabled check.
    """
    monkeypatch.setitem(packages._PACKAGE_OPTS, "check_l4t_version", False)

    good = {"name": "good", "requires": [], "cuda_libs": ["cudnn"]}
    bad = {"name": "bad", "requires": [], "cuda_libs": ["made_up_lib"]}

    assert packages.validate_package(good) == [good]
    assert packages.validate_package(bad) == []


# ---------------------------------------------------------------------------
# parse_yaml_header: a #--- Dockerfile header with cuda_libs parses through,
# and find_package(pkg)['cuda_libs'] == ['cudnn', 'nccl']
# ---------------------------------------------------------------------------

_DOCKERFILE_HEADER = """\
#---
# name: my-package
# cuda_libs: [cudnn, nccl]
#---
FROM scratch
"""


def test_parse_yaml_header_parses_cuda_libs_list(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(_DOCKERFILE_HEADER)

    config = packages.parse_yaml_header(str(dockerfile))

    assert config is not None, "header should not be silently dropped"
    assert config["cuda_libs"] == ["cudnn", "nccl"]


def test_parse_yaml_header_with_unknown_cuda_lib_disables_not_silently_drops(
    tmp_path, capsys
):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "#---\n"
        "# name: my-package\n"
        "# cuda_libs: [cudnn, made_up_lib]\n"
        "#---\n"
        "FROM scratch\n"
    )

    # apply_config() is what actually calls validate_lists() on a header
    # dict; parse_yaml_header() itself only calls validate_dict() (the
    # unknown-*key* path), so exercise the unknown-*value* path the same
    # way the real scan does: through apply_config().
    config = packages.parse_yaml_header(str(dockerfile))
    assert config is not None

    package = {"name": "my-package", "path": str(tmp_path)}
    packages.apply_config(package, config)

    assert package["disabled"] is True
    out = capsys.readouterr()
    message = out.out + out.err
    assert "my-package" in message
    assert "made_up_lib" in message


def test_find_package_with_cuda_libs_dict_roundtrips():
    """
    A package dict handed directly to find_package() (which short-circuits
    through validate_dict() when given a dict rather than a name/pattern)
    round-trips its cuda_libs list unchanged.
    """
    pkg = {"name": "my-package", "cuda_libs": ["cudnn", "nccl"]}
    found = packages.find_package(pkg)
    assert found is pkg
    assert found["cuda_libs"] == ["cudnn", "nccl"]
