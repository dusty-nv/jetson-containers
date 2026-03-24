# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

jetson-containers is a modular container build system for NVIDIA Jetson devices. It discovers packages, resolves their dependencies, chains their Dockerfiles together, and runs `docker build` to produce multi-stage images optimized for specific L4T/JetPack/CUDA versions.

## Commands

### Build & Run
```bash
# Build one or more packages (chains Dockerfiles in dependency order)
jetson-containers build <package> [<package2> ...]
# or equivalently:
./build.sh <package> [<package2> ...]

# Find and run a compatible container image
jetson-containers run $(autotag <package>) [command]
# or equivalently:
./run.sh $(autotag <package>) [command]

# Find the best matching image tag (local → DockerHub → build)
autotag <package>
```

### Code Style
```bash
# Install pre-commit hooks (one-time setup)
pip install -r requirements.txt
pre-commit install

# Run linting/formatting checks manually
pre-commit run --all-files
```

Pre-commit runs **Black** (formatter, line-length=88) and **Flake8** (linter) on `jetson_containers/`, `packages/example/`, and `test_precommit.py`. Note: `skip-string-normalization = true` in `pyproject.toml`.

### Installation
```bash
bash install.sh   # sets up venv, installs deps, symlinks CLI tools to /usr/local/bin
```

## Architecture

### Package System

Every buildable unit is a **package**. Packages live under `packages/<category>/<name>/` and define metadata via one of (in priority order):

1. **YAML header in Dockerfile** — metadata between `#---` markers at the top
2. **`config.yaml` / `config.yml`** — static YAML config file
3. **`config.json`** — meta-container (no Dockerfile, just dependency composition)
4. **`config.py`** — dynamic config, returns a list of package dicts at build time

Key metadata fields: `name`, `alias`, `depends` (build-order deps), `requires` (version constraints on L4T/CUDA/Python), `test`, `build_args`, `dockerfile`.

### Build Pipeline

1. **`jetson_containers/packages.py`** — scans `packages/` recursively, runs `config.py` scripts, resolves the dependency graph, filters by L4T version compatibility, and exposes a custom Python meta path finder so packages can `from packages.ml.pytorch.version import ...`
2. **`jetson_containers/build.py`** — CLI entry point; parses args and dispatches
3. **`jetson_containers/container.py`** — core build logic: stitches multiple package Dockerfiles into one combined `Dockerfile`, invokes `docker build`, optionally runs per-package tests, and optionally pushes to a registry
4. **`jetson_containers/l4t_version.py`** — detects architecture (tegra-aarch64, aarch64, x86_64) and reads L4T/JetPack/CUDA/GPU-arch info from `/etc/nv_tegra_release`; all version variables (e.g. `CUDA_VERSION`, `PYTORCH_VERSION`) flow from here into build args

### Dynamic Package Generation

`config.py` files execute at build time and can return multiple package dicts to produce version variants. For example, PyTorch's `config.py` generates `pytorch:2.8`, `pytorch:2.8-all`, `pytorch:2.8-builder`, etc. Aliases (`torch` → `pytorch:2.8`) are resolved during package discovery.

### Version Overrides

Copy `.env.default` → `.env` to override `L4T_VERSION`, `CUDA_VERSION`, `LSB_RELEASE`, `CUDA_ARCH`, local PyPI/APT mirror URLs, and SCP/webhook credentials without modifying tracked files.

### Creating a New Package

1. Create `packages/<category>/<name>/Dockerfile` with a `#---` YAML header (or a `config.yaml`/`config.py`).
2. Declare `depends` on upstream packages (e.g., `pytorch`, `cuda`).
3. Optionally add `test.py` or `test.sh` for post-build validation.
4. Build: `jetson-containers build <name>`

### Key Supporting Modules

- `jetson_containers/tag.py` — image tagging conventions
- `jetson_containers/docs.py` — auto-generates per-package docs from metadata
- `jetson_containers/ci.py` — CI/CD helpers
- `jetson_containers/network.py` / `webhook.py` — GitHub API integration and build notifications
- `jetson_containers/logging.py` — colored terminal output used throughout
