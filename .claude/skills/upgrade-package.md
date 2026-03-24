# Upgrade Package Version

Help the user upgrade the version of one or more packages in this jetson-containers repository.

## What the user will provide

The user will tell you the package name and the new version. If they only give you the package name, ask for the target version before proceeding.

## How version definitions work

Each package under `packages/<category>/<name>/` defines its version in one of these ways:

1. **`version.py`** — a standalone file that exports a version constant, e.g. `PYTORCH_VERSION = Version('2.8')`. This is the canonical version used by `config.py`.
2. **`config.py`** — calls a helper function (like `pytorch_pip(version, requires=...)`) with explicit version strings in a `package = [...]` list. Adding a new version = adding a new line to this list.
3. **`config.yaml` / YAML header in Dockerfile** — static `build_args` with version pinned directly.
4. **`Dockerfile` `ARG` lines** — `ARG SOME_VERSION=x.y.z`.

## Step-by-step workflow

1. **Locate the package directory**: search under `packages/` for the package name.
2. **Read all config files** in that directory (`config.py`, `config.yaml`, `version.py`, `Dockerfile`, `Dockerfile.*`).
3. **Identify where the version is defined** (see above). Show the user the current version(s) before making changes.
4. **Apply the upgrade**:
   - If there is a `version.py`, update the version constant there first; `config.py` will pick it up automatically.
   - If `config.py` uses a list of `pytorch_pip(version, ...)` calls, **add** a new entry for the new version (do not remove old ones — the repo keeps historical versions for different JetPack targets). Set the new version as the latest by updating the `PYTORCH_VERSION`-like constant in `version.py`.
   - If the version is in a YAML `build_args` or a Dockerfile `ARG`, do a targeted edit of that value.
5. **Check for cascading dependencies**: grep for the old version string across other config files in the same package directory. Common locations: `Dockerfile` `ARG` defaults, `build_args` dicts, `requires` strings, download URLs, wheel filenames.
6. **Update download artifacts if needed**: if the Dockerfile fetches a wheel, tarball, or pre-built binary, update the URL/filename to match the new version. Check the upstream source (PyPI, NVIDIA, GitHub releases) if the user hasn't provided a URL.
7. **Show a diff summary** of every file you changed before finishing.

## Things to preserve

- Do **not** delete existing version entries in `config.py` lists — old JetPack targets still need them.
- Keep the `requires` constraints on old entries unchanged.
- Do not change `alias` lists for older versions.
- Follow the existing code style (no reformatting unrelated code, no added comments).

## When to ask vs. proceed

- Ask for the target version if not given.
- Ask about JetPack `requires` constraint only if the new version introduces a minimum JetPack requirement that differs from the previous latest entry.
- Proceed with everything else silently and show the diff at the end.
