"""
Tests for the chain-aware cudastack wiring in jetson_containers/container.py
(task t3 of the cudastack-chain-aware plan; covers c3, h3, c12, h10, h24).

build_container() is driven end-to-end with simulate=True against a small
fake package registry: resolve_dependencies / find_package / tag_container /
test_container / logging are replaced on the loaded container module, and the
docker build command of every stage is captured from log_block(). Nothing
touches docker, the network, or the real package scan.

container.py is imported the same way tests/test_packages_cuda_libs.py
imports packages.py: a placeholder "jetson_containers" package is put in
sys.modules so jetson_containers/__init__.py (and its check_dependencies()
pip/network side effects) never runs. dockerhub_api / tabulate / termcolor
are display- or push-only imports; stand-ins are registered only if they
are not installed.
"""

import copy
import importlib
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_DIR = _REPO_ROOT / "jetson_containers"


def _stub_module(name, **attrs):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
    except ImportError:
        stub = types.ModuleType(name)
        for key, val in attrs.items():
            setattr(stub, key, val)
        sys.modules[name] = stub


def _load_container_module():
    _stub_module("tabulate", tabulate=lambda *a, **k: "")
    _stub_module(
        "termcolor",
        colored=lambda text, *a, **k: text,
        cprint=lambda text, *a, **k: print(text),
    )
    _stub_module("dockerhub_api")

    if "jetson_containers" not in sys.modules:
        fake_pkg = types.ModuleType("jetson_containers")
        fake_pkg.__path__ = [str(_PKG_DIR)]
        sys.modules["jetson_containers"] = fake_pkg

    return importlib.import_module("jetson_containers.container")


container = _load_container_module()


# ---------------------------------------------------------------------------
# fake registry + simulated build harness
# ---------------------------------------------------------------------------

STANDARD_WITH = {
    'WITH_CUDNN': '1',
    'WITH_TENSORRT': '1',
    'WITH_NCCL': '1',
    'WITH_CUDSS': '1',
    'WITH_CUSPARSELT': '1',
    'WITH_CUTENSOR': '1',
    'WITH_GDRCOPY': '0',
    'WITH_NVPL': '0',
    'WITH_NVSHMEM': '1',
}


def _cudastack_build_args():
    # non-WITH_* keys interleaved before/after, the way cuda_stack_args()
    # + the preset toggles produce them; these must pass through unchanged.
    args = {
        'CUDNN_VERSION': '9.10.2',
        'TENSORRT_URL': 'https://example.invalid/trt.tar.gz',
        'NCCL_VERSION': '2.27.7',
        'CUDA_INSTALLED_VERSION': 130,
    }
    args.update(STANDARD_WITH)
    return args


def _pkg(name, **extra):
    pkg = {
        'name': name,
        'path': f'/fake/packages/{name.split(":")[0]}',
        'dockerfile': 'Dockerfile',
    }
    pkg.update(extra)
    return pkg


def make_registry():
    # 'depends' on 'cudastack:standard' models the realistic shape (e.g.
    # packages/code/vscode/config.py: `depends=['cudastack']`): a package that
    # actually uses cudastack declares it, directly or transitively. Packages
    # after cudastack in a chain that DON'T reach it via 'depends' (no path in
    # their resolve_dependencies() closure) are the "doesn't use cudastack at
    # all" case the chain-aware narrowing must ignore (see the jax fixture
    # below: numpy/cmake/onnx/llvm don't depend on cudastack).
    return {
        'cuda': _pkg('cuda', build_args={'CUDA_VERSION': '13.0'}),
        'cudastack:standard': _pkg(
            'cudastack:standard',
            build_args=_cudastack_build_args(),
            depends=['cuda'],
        ),
        'python': _pkg('python', build_args={'PYTHON_VERSION': '3.12'}),
        'undeclared': _pkg('undeclared', depends=['cudastack:standard']),
        'wants-nccl': _pkg(
            'wants-nccl', cuda_libs=['nccl'], depends=['cudastack:standard']
        ),
        'wants-nothing': _pkg(
            'wants-nothing', cuda_libs=[], depends=['cudastack:standard']
        ),
        'wants-cutensor': _pkg(
            'wants-cutensor',
            cuda_libs=['cutensor'],
            build_args={'FOO': 'bar'},
            depends=['cudastack:standard'],
        ),
    }


class Harness:
    """Run build_container(simulate=True) against a fake registry."""

    def __init__(self, monkeypatch, tmp_path, registry=None):
        self.registry = registry if registry is not None else make_registry()
        self.blocks = []
        self.infos = []
        self.chain = None
        self._resolve_calls = 0
        # package names whose per-package closure query (the call
        # _cudastack_stage() makes as resolve_dependencies([p], ...)) must
        # raise instead of resolving, to exercise the "resolve failure
        # doesn't crash the build" fix.
        self.unresolvable = set()
        # skip_packages seen on every resolve_dependencies call after the
        # first (i.e. every per-package closure query _cudastack_stage()
        # makes), so tests can assert it was forwarded.
        self.resolve_skip_packages_seen = []
        log_dir = tmp_path / 'logs' / 'build'
        log_dir.mkdir(parents=True)

        m = monkeypatch
        m.setattr(container, 'resolve_dependencies', self.resolve_dependencies)
        m.setattr(container, 'find_package', self.find_package)
        m.setattr(container, 'get_log_dir', lambda *a, **k: str(log_dir))
        m.setattr(container, 'tag_container', lambda *a, **k: None)
        m.setattr(container, 'test_container', lambda *a, **k: None)
        m.setattr(container, 'get_jetpack_version', lambda: '7.0')
        m.setattr(container, 'sudo_prefix', lambda *a, **k: '')
        # dusty's build_container() opens each stage's Dockerfile to probe
        # for BuildKit `RUN --device` usage; the fake packages have none.
        m.setattr(
            container,
            '_dockerfile_requests_buildkit_device',
            lambda *a, **k: False,
        )
        m.setattr(
            container,
            'log_block',
            lambda title, cmd, **k: self.blocks.append(cmd),
        )
        m.setattr(container, 'log_status', lambda *a, **k: None)
        m.setattr(container, 'log_success', lambda *a, **k: None)
        m.setattr(
            container,
            'log_info',
            lambda text, **k: self.infos.append(str(text)),
        )
        m.delenv('CUDASTACK_CHAIN_AWARE', raising=False)
        m.delenv('APT_PROXY', raising=False)
        m.delenv('BUILD_DELAY', raising=False)

    def find_package(self, package):
        return self.registry[package]

    def resolve_dependencies(self, packages, skip_packages=[]):
        """
        Fake resolve_dependencies(). The FIRST call in any build() is the
        top-level call build_container() makes to resolve the requested
        packages into the build chain -- tests construct that chain directly
        (it doesn't necessarily match what real 'depends' resolution would
        produce, e.g. artificial before/after orderings), so it returns the
        chain unchanged. Every call after that is a closure query made by
        _cudastack_stage() for a single package above cudastack, and is
        answered for real by walking the fake registry's 'depends' lists --
        this is what lets the chain-aware filter see whether a package's own
        dependency closure reaches a cudastack package.
        """
        self._resolve_calls += 1
        if self._resolve_calls == 1:
            return list(self.chain)

        self.resolve_skip_packages_seen.append(list(skip_packages))

        if len(packages) == 1 and packages[0] in self.unresolvable:
            raise KeyError(f"couldn't find package: {packages[0]}")

        result = []

        def visit(pkg_name):
            if pkg_name in result:
                return
            result.append(pkg_name)
            for dep in self.registry[pkg_name].get('depends', []):
                visit(dep)

        for p in packages:
            visit(p)
        return result

    def build(
        self, chain, name='test:r1', build_args=None, skip_packages=None
    ):
        self.chain = chain
        self.blocks = []
        self.infos = []
        self._resolve_calls = 0
        self.resolve_skip_packages_seen = []
        container.build_container(
            name=name,
            packages=[chain[-1]],
            base='base:img',
            build_args=build_args,
            simulate=True,
            skip_tests=['all'],
            skip_packages=skip_packages if skip_packages is not None else [],
            # plain `docker build` command shape: no BuildKit Dockerfile
            # rewriting (ccache/device) against the fake package paths
            buildkit=False,
            ccache=False,
        )
        assert len(self.blocks) == len(chain)
        return dict(zip(chain, self.blocks))


# dusty's build_container() adds this constant build-arg to every stage,
# right after BASE_IMAGE; it is not part of what these tests compare.
_DUSTY_CONSTANT_ARGS = ('--build-arg NVIDIA_DRIVER_CAPABILITIES=all',)


def build_arg_lines(cmd):
    return [
        line.strip()
        for line in cmd.split('\n')
        if '--build-arg' in line
        and not line.strip().startswith(_DUSTY_CONSTANT_ARGS)
    ]


def tag_of(cmd):
    for line in cmd.split('\n'):
        line = line.strip()
        if line.startswith('--tag '):
            return line[len('--tag '):].rstrip(' \\')
    raise AssertionError('no --tag in cmd')


def with_value(cmd, key):
    """The quoted per-package value of a WITH_* build arg in a stage cmd."""
    values = [
        line
        for line in build_arg_lines(cmd)
        if line.startswith(f'--build-arg {key}="')
    ]
    assert len(values) == 1, values
    return values[0].split('=', 1)[1].split('"')[1]


def expected_preset_lines():
    return [
        f'--build-arg {k}="{v}" \\' for k, v in _cudastack_build_args().items()
    ]


@pytest.fixture
def harness(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


# ---------------------------------------------------------------------------
# c3 / h3: the cudastack stage's WITH_* come from the packages AFTER it
# ---------------------------------------------------------------------------


def test_declared_chain_narrows_cudastack_with_args(harness):
    cmds = harness.build(
        ['cuda', 'cudastack:standard', 'wants-nccl', 'wants-nothing']
    )
    cmd = cmds['cudastack:standard']

    assert with_value(cmd, 'WITH_CUDNN') == '1'
    assert with_value(cmd, 'WITH_NCCL') == '1'
    for key in (
        'WITH_TENSORRT',
        'WITH_CUDSS',
        'WITH_CUSPARSELT',
        'WITH_CUTENSOR',
        'WITH_NVSHMEM',
        'WITH_GDRCOPY',
        'WITH_NVPL',
    ):
        assert with_value(cmd, key) == '0', key


def test_declared_chain_keeps_non_with_args_unchanged_and_in_order(harness):
    cmds = harness.build(['cuda', 'cudastack:standard', 'wants-nccl'])
    lines = build_arg_lines(cmds['cudastack:standard'])
    keys = [line.split(' ', 1)[1].split('=', 1)[0] for line in lines]
    assert keys == ['BASE_IMAGE'] + list(_cudastack_build_args().keys())
    assert '--build-arg CUDNN_VERSION="9.10.2" \\' in lines
    assert (
        '--build-arg TENSORRT_URL="https://example.invalid/trt.tar.gz" \\'
        in lines
    )
    assert '--build-arg CUDA_INSTALLED_VERSION="130" \\' in lines


def test_declared_chain_gets_truthful_stage_tag(harness):
    cmds = harness.build(['cuda', 'cudastack:standard', 'wants-nccl'])
    assert tag_of(cmds['cudastack:standard']) == 'test:r1-cudastack_cudnn-nccl'
    # the next stage is built FROM the narrowed stage
    assert (
        '--build-arg BASE_IMAGE=test:r1-cudastack_cudnn-nccl \\'
        in build_arg_lines(cmds['wants-nccl'])
    )


def test_simulate_logs_one_line_naming_the_computed_set(harness):
    harness.build(['cuda', 'cudastack:standard', 'wants-nccl'])
    lines = [
        line
        for line in harness.infos
        if 'cudastack' in line and 'chain' in line
    ]
    assert len(lines) == 1
    assert 'cudnn' in lines[0] and 'nccl' in lines[0]
    assert 'tensorrt' not in lines[0]


def test_only_packages_after_cudastack_count(harness):
    # 'wants-cutensor' sits BEFORE cudastack in this (artificial) chain; it
    # must not widen the set. Only 'wants-nccl' (after) counts.
    cmds = harness.build(
        ['wants-cutensor', 'cuda', 'cudastack:standard', 'wants-nccl']
    )
    assert with_value(cmds['cudastack:standard'], 'WITH_CUTENSOR') == '0'
    assert with_value(cmds['cudastack:standard'], 'WITH_NCCL') == '1'


# ---------------------------------------------------------------------------
# o2: every other stage's docker build command is byte-identical
# ---------------------------------------------------------------------------


def test_other_stages_are_byte_identical_to_opt_out(harness, monkeypatch):
    chain = [
        'python',
        'cuda',
        'cudastack:standard',
        'wants-nccl',
        'wants-cutensor',
    ]
    aware = harness.build(chain)
    monkeypatch.setenv('CUDASTACK_CHAIN_AWARE', '0')
    legacy = harness.build(chain)

    for pkg in ('python', 'cuda', 'wants-cutensor'):
        assert aware[pkg] == legacy[pkg], pkg

    # the stage right after cudastack differs ONLY in its BASE_IMAGE line
    diff = [
        (a, b)
        for a, b in zip(
            aware['wants-nccl'].split('\n'), legacy['wants-nccl'].split('\n')
        )
        if a != b
    ]
    assert diff == [
        (
            '  --build-arg BASE_IMAGE='
            'test:r1-cudastack_cudnn-cutensor-nccl \\',
            '  --build-arg BASE_IMAGE=test:r1-cudastack_standard \\',
        )
    ]


# ---------------------------------------------------------------------------
# byte-identical to today: undeclared-only chain, bare cudastack, opt-out
# ---------------------------------------------------------------------------


def _assert_todays_cudastack(cmd):
    assert tag_of(cmd) == 'test:r1-cudastack_standard'
    assert (
        build_arg_lines(cmd)
        == ['--build-arg BASE_IMAGE=test:r1-cuda \\'] + expected_preset_lines()
    )


def test_undeclared_only_chain_is_todays_command(harness):
    cmds = harness.build(['cuda', 'cudastack:standard', 'undeclared'])
    _assert_todays_cudastack(cmds['cudastack:standard'])


def test_mixed_chain_with_an_undeclared_package_keeps_preset(harness):
    cmds = harness.build(
        ['cuda', 'cudastack:standard', 'wants-nccl', 'undeclared']
    )
    _assert_todays_cudastack(cmds['cudastack:standard'])


def test_bare_cudastack_build_is_todays_command(harness):
    cmds = harness.build(['cuda', 'cudastack:standard'])
    _assert_todays_cudastack(cmds['cudastack:standard'])


@pytest.mark.parametrize('value', ['0', 'false', 'off', 'OFF', 'False'])
def test_env_opt_out_restores_preset_and_todays_tag(
    harness, monkeypatch, value
):
    monkeypatch.setenv('CUDASTACK_CHAIN_AWARE', value)
    cmds = harness.build(['cuda', 'cudastack:standard', 'wants-nccl'])
    _assert_todays_cudastack(cmds['cudastack:standard'])


def test_build_args_opt_out_restores_preset_and_todays_tag(harness):
    cmds = harness.build(
        ['cuda', 'cudastack:standard', 'wants-nccl'],
        build_args={'CUDASTACK_CHAIN_AWARE': '0'},
    )
    cmd = cmds['cudastack:standard']
    assert tag_of(cmd) == 'test:r1-cudastack_standard'
    assert build_arg_lines(cmd) == [
        '--build-arg BASE_IMAGE=test:r1-cuda \\'
    ] + expected_preset_lines() + ['--build-arg CUDASTACK_CHAIN_AWARE=0 \\']


def test_env_on_value_keeps_narrowing(harness, monkeypatch):
    monkeypatch.setenv('CUDASTACK_CHAIN_AWARE', '1')
    cmds = harness.build(['cuda', 'cudastack:standard', 'wants-nccl'])
    assert tag_of(cmds['cudastack:standard']) == 'test:r1-cudastack_cudnn-nccl'


# ---------------------------------------------------------------------------
# c12 / h10: the shared registry is never mutated
# ---------------------------------------------------------------------------


def test_registry_build_args_not_mutated(harness):
    before = copy.deepcopy(harness.registry)
    harness.build(['cuda', 'cudastack:standard', 'wants-nccl'])
    assert harness.registry == before


def test_two_chains_in_one_process_do_not_leak(harness):
    first = harness.build(['cuda', 'cudastack:standard', 'wants-nccl'])
    second = harness.build(['cuda', 'cudastack:standard', 'undeclared'])
    third = harness.build(['cuda', 'cudastack:standard', 'wants-cutensor'])

    assert (
        tag_of(first['cudastack:standard']) == 'test:r1-cudastack_cudnn-nccl'
    )
    _assert_todays_cudastack(second['cudastack:standard'])
    assert (
        tag_of(third['cudastack:standard'])
        == 'test:r1-cudastack_cudnn-cutensor'
    )
    assert with_value(third['cudastack:standard'], 'WITH_NCCL') == '0'


# ---------------------------------------------------------------------------
# o3 / h24: global --build-args WITH_X still wins; tag = effective set
# ---------------------------------------------------------------------------


def test_global_with_override_is_appended_last_and_reflected_in_tag(harness):
    cmds = harness.build(
        ['cuda', 'cudastack:standard', 'wants-nccl'],
        build_args={'WITH_TENSORRT': '1'},
    )
    cmd = cmds['cudastack:standard']
    lines = build_arg_lines(cmd)

    assert (
        with_value(cmd, 'WITH_TENSORRT') == '0'
    )  # computed per-package value
    assert (
        lines[-1] == '--build-arg WITH_TENSORRT=1 \\'
    )  # global override, last wins
    assert tag_of(cmd) == 'test:r1-cudastack_cudnn-nccl-tensorrt'


def test_global_override_turning_off_a_lib_is_reflected_in_tag(harness):
    cmds = harness.build(
        ['cuda', 'cudastack:standard', 'wants-nccl'],
        build_args={'WITH_NCCL': '0'},
    )
    assert tag_of(cmds['cudastack:standard']) == 'test:r1-cudastack_cudnn'


def test_global_override_equal_to_computed_keeps_tag(harness):
    cmds = harness.build(
        ['cuda', 'cudastack:standard', 'wants-nccl'],
        build_args={'WITH_NCCL': '1'},
    )
    assert tag_of(cmds['cudastack:standard']) == 'test:r1-cudastack_cudnn-nccl'


# ---------------------------------------------------------------------------
# Docker tag length budget (<= 128 chars after the ':')
# ---------------------------------------------------------------------------


def _tag_part(image):
    last = image.rsplit('/', 1)[-1]
    return last.split(':', 1)[1] if ':' in last else last


@pytest.mark.parametrize(
    'name',
    [
        'repo:' + 'v' * 100,  # explicit tag
        'registry:5000/ns/repo:' + 'x' * 110,  # registry port + explicit tag
        'ns/' + 'r' * 100,  # no tag: the stage name is the repo
    ],
)
def test_long_names_keep_stage_tag_within_128(harness, name):
    registry = harness.registry
    registry['many'] = _pkg(
        'many',
        cuda_libs=['nccl', 'cudss', 'cusparselt', 'nvshmem'],
        depends=['cudastack:standard'],
    )
    cmds = harness.build(['cuda', 'cudastack:standard', 'many'], name=name)
    tag = tag_of(cmds['cudastack:standard'])
    assert tag.startswith(name + '-cudastack_')
    assert len(_tag_part(tag)) <= 128
    # still narrowed (not silently the preset)
    assert with_value(cmds['cudastack:standard'], 'WITH_TENSORRT') == '0'


# ---------------------------------------------------------------------------
# deviation d8: only cudastack DEPENDENTS (own resolve_dependencies() closure
# reaches a cudastack package) count toward the chain-aware set. A package
# after cudastack that doesn't use it at all (no path to it in its closure)
# must not force the full preset just by being undeclared.
# ---------------------------------------------------------------------------


def _jax_registry():
    # models the real jax chain (build-essential, pip_cache:cu130, cuda,
    # python, cudastack:standard, numpy, cmake, onnx, llvm:21, jax): only jax
    # itself depends on cudastack; numpy/cmake/onnx/llvm:21 are undeclared
    # AND don't use cudastack at all.
    registry = make_registry()
    registry.update(
        {
            'numpy': _pkg('numpy', depends=['python']),
            'cmake': _pkg('cmake'),
            'onnx': _pkg('onnx'),
            'llvm:21': _pkg('llvm:21'),
            'jax': _pkg(
                'jax',
                cuda_libs=['cudnn', 'nccl'],
                depends=[
                    'cuda',
                    'cudastack:standard',
                    'numpy',
                    'onnx',
                    'llvm:21',
                ],
            ),
        }
    )
    return registry


def test_jax_chain_narrows_to_only_the_dependent_packages_declared_libs(
    monkeypatch, tmp_path
):
    harness = Harness(monkeypatch, tmp_path, registry=_jax_registry())
    cmds = harness.build(
        [
            'cuda',
            'cudastack:standard',
            'numpy',
            'cmake',
            'onnx',
            'llvm:21',
            'jax',
        ]
    )
    cmd = cmds['cudastack:standard']

    assert tag_of(cmd) == 'test:r1-cudastack_cudnn-nccl'
    assert with_value(cmd, 'WITH_CUDNN') == '1'
    assert with_value(cmd, 'WITH_NCCL') == '1'
    for key in (
        'WITH_TENSORRT',
        'WITH_CUDSS',
        'WITH_CUSPARSELT',
        'WITH_CUTENSOR',
        'WITH_NVSHMEM',
        'WITH_GDRCOPY',
        'WITH_NVPL',
    ):
        assert with_value(cmd, key) == '0', key


def test_crossformer_like_chain_of_undeclared_dependents_keeps_full_preset(
    monkeypatch, tmp_path
):
    # pytorch and opencv actually depend on cudastack (like the real
    # packages/robots/ros/config.py depends list) but are undeclared, so they
    # force the full preset -- same as today's no-narrowing command -- even
    # though jax (declared) is also in the chain.
    registry = _jax_registry()
    registry.update(
        {
            'pytorch': _pkg('pytorch', depends=['cudastack:standard']),
            'opencv': _pkg('opencv', depends=['cudastack:standard']),
        }
    )
    harness = Harness(monkeypatch, tmp_path, registry=registry)
    cmds = harness.build(
        ['cuda', 'cudastack:standard', 'pytorch', 'opencv', 'jax']
    )
    # _assert_todays_cudastack pins the exact tag + build-arg lines that every
    # other "full preset" chain in this file produces (e.g.
    # test_undeclared_only_chain_is_todays_command), i.e. byte-identical to
    # the no-declaration case.
    _assert_todays_cudastack(cmds['cudastack:standard'])


def test_only_package_after_cudastack_not_depending_on_it_keeps_full_preset(
    harness,
):
    # 'numpy' sits after cudastack but its own dependency closure never
    # reaches cudastack -- it must be dropped from consideration entirely,
    # leaving an empty chain_above (compute_set's empty-chain rule: full
    # preset), not treated as an "undeclared dependent".
    registry = harness.registry
    registry['numpy'] = _pkg('numpy', depends=['python'])
    cmds = harness.build(['cuda', 'cudastack:standard', 'numpy'])
    _assert_todays_cudastack(cmds['cudastack:standard'])


def test_transitive_dependent_of_cudastack_counts_and_forces_full_preset(
    harness,
):
    # X depends on pytorch, which depends on cudastack -- X itself has no
    # direct 'depends' on cudastack, only a transitive one. X declares
    # nothing, so (since it counts) it forces the full preset.
    registry = harness.registry
    registry['pytorch'] = _pkg('pytorch', depends=['cudastack:standard'])
    registry['x'] = _pkg('x', depends=['pytorch'])
    cmds = harness.build(['cuda', 'cudastack:standard', 'x'])
    _assert_todays_cudastack(cmds['cudastack:standard'])


def test_transitive_dependent_of_cudastack_that_is_declared_narrows(harness):
    # same transitive shape as above, but X IS declared (cuda_libs=['nccl']):
    # proves the closure walk is genuinely transitive (X has no direct
    # 'depends' on cudastack) rather than a shallow direct-depends check --
    # if X were wrongly excluded, chain_above would be empty and this would
    # produce the full preset instead of narrowing.
    registry = harness.registry
    registry['pytorch'] = _pkg('pytorch', depends=['cudastack:standard'])
    registry['x'] = _pkg('x', depends=['pytorch'], cuda_libs=['nccl'])
    cmds = harness.build(['cuda', 'cudastack:standard', 'x'])
    cmd = cmds['cudastack:standard']
    assert tag_of(cmd) == 'test:r1-cudastack_cudnn-nccl'
    assert with_value(cmd, 'WITH_NCCL') == '1'
    assert with_value(cmd, 'WITH_TENSORRT') == '0'


def test_cudastack_variant_itself_is_never_above(harness):
    # a second cudastack variant appearing after the built one in the chain
    # must not count as a "dependent" of itself.
    registry = harness.registry
    registry['cudastack:minimal'] = _pkg('cudastack:minimal', depends=['cuda'])
    cmds = harness.build(['cuda', 'cudastack:standard', 'cudastack:minimal'])
    _assert_todays_cudastack(cmds['cudastack:standard'])


# ---------------------------------------------------------------------------
# review fix (item 2): skip_packages must be forwarded to _cudastack_stage's
# per-package resolve_dependencies() calls, and a resolve failure for one
# package must not crash the build.
# ---------------------------------------------------------------------------


def test_skip_packages_is_forwarded_to_per_package_resolve_calls(harness):
    cmds = harness.build(
        ['cuda', 'cudastack:standard', 'wants-nccl'],
        skip_packages=['torchsde', 'some-other-skip'],
    )
    assert cmds  # build still produced stages
    # every per-package closure query _cudastack_stage() made saw the same
    # skip_packages the top-level build was given
    assert harness.resolve_skip_packages_seen
    for seen in harness.resolve_skip_packages_seen:
        assert seen == ['torchsde', 'some-other-skip']


def test_default_empty_skip_packages_is_forwarded(harness):
    harness.build(['cuda', 'cudastack:standard', 'wants-nccl'])
    assert harness.resolve_skip_packages_seen
    for seen in harness.resolve_skip_packages_seen:
        assert seen == []


def test_resolve_failure_for_one_package_does_not_raise(harness):
    # reproduces the operator's report: resolve_dependencies(['comfyui'],
    # skip_packages=['torchsde']) can still raise for a package whose
    # dependency isn't covered by skip_packages -- the build must not crash.
    registry = harness.registry
    registry['comfyui'] = _pkg('comfyui', depends=['cudastack:standard'])
    harness.unresolvable.add('comfyui')

    cmds = harness.build(
        ['cuda', 'cudastack:standard', 'comfyui'], skip_packages=['torchsde']
    )
    assert 'cudastack:standard' in cmds  # did not raise


def test_resolve_failure_treats_package_as_cudastack_dependent(harness):
    # an unresolvable package is treated conservatively: since it counts as
    # a dependent and declares nothing (undeclared), it forces the full
    # preset -- same as any other undeclared dependent in this file.
    registry = harness.registry
    registry['comfyui'] = _pkg('comfyui', depends=['cudastack:standard'])
    harness.unresolvable.add('comfyui')

    cmds = harness.build(
        ['cuda', 'cudastack:standard', 'comfyui'], skip_packages=['torchsde']
    )
    _assert_todays_cudastack(cmds['cudastack:standard'])


def test_resolve_failure_declared_package_still_narrows(harness):
    # same unresolvable-closure shape, but the package IS declared
    # (cuda_libs=['nccl']) -- it still counts (conservative treatment) and
    # contributes its own declaration, narrowing rather than forcing the
    # full preset.
    registry = harness.registry
    registry['comfyui'] = _pkg(
        'comfyui', depends=['cudastack:standard'], cuda_libs=['nccl']
    )
    harness.unresolvable.add('comfyui')

    cmds = harness.build(
        ['cuda', 'cudastack:standard', 'comfyui'], skip_packages=['torchsde']
    )
    cmd = cmds['cudastack:standard']
    assert tag_of(cmd) == 'test:r1-cudastack_cudnn-nccl'
    assert with_value(cmd, 'WITH_NCCL') == '1'
    assert with_value(cmd, 'WITH_TENSORRT') == '0'
