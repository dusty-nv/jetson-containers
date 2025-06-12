# Package Definitions

A package is one building block of a container - typically composed of a Dockerfile and optional configuration scripts.

You might notice that the Dockerfiles in this repo have special package metadata encoded in their header comments:

```dockerfile
#---
# name: pytorch
# alias: torch
# group: ml
# config: config.py
# depends: [python, numpy, onnx]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

...
```

The text between `#---` is YAML and is extracted by the build system.  Each package dict has the following keys:

| Key           |         Type         | Description                                                                             |
|---------------|:--------------------:|-----------------------------------------------------------------------------------------|
| `name`        |         `str`        | the name of the package                                                                 |
| `alias`       | `str` or `list[str]` | alternate names the package can be referred to by                                       |
| `build_args`  |        `dict`        | `ARG:VALUE` pairs that are `--build-args` to `docker build`                             |
| `build_flags` |         `str`        | additional options that get added to the `docker build` command                         |
| `config`      | `str` or `list[str]` | one or more config files to load (`.py`, `.json`, `.yml`, `.yaml`)                      |
| `depends`     | `str` or `list[str]` | list of packages that this package depends on, and will be built                        |
| `disabled`    | `bool`               | set to `true` for the package to be disabled                                            |
| `dockerfile`  | `str`                | filename of the Dockerfile (optional)                                                   |
| `docs`        | `str`                | text or markdown that is added to a package's auto-generated readme                     |
| `group`       | `str`                | optional group the package belongs to (e.g. `ml`, `llm`, `cuda`)                        |
| `notes`       | `str`                | brief one-line docs that are added to a package's readme table                          |
| `path`        | `str`                | path to the package's directory (automatically populated)                               |
| `prefix`      | `str`                | text prepended to the container tag. not part of the package's name for referencing.    |
| `postfix`     | `str`                | text appended to the container tag (default is `r$L4T_VERSION`)                         |
| `requires`    | `str` or `list[str]` | the version(s) of L4T or CUDA the package is compatible with (e.g. `>=35.2.1` for JetPack 5.1+) |
| `test`        | `str` or `list[str]` | one or more test commands/scripts to run (`.py`, `.sh`, or a shell command)             |

> * These keys can all be accessed by any of the configuration methods below<br>
> * Any filenames or paths should be relative to the package's `path`<br>
> * See the [Version Specifiers Specification](https://packaging.pypa.io/en/latest/specifiers.html) for valid syntax around `requires`
> * `requires` can also check for CUDA version (`>=cu124`) and Python version (`>=py310`) and can be a list (`['>=r36', '>=cu122']`)

Packages can also include nested sub-packages (for example, all the [ROS variants](/packages/ros)), which are typically generated in a config file.

## YAML

In lieu of having the package metadata right there in the Dockerfile header, packages can provide a separate YAML file (normally called `config.yaml` or `config.yml`) with the same information:

```yaml
name: pytorch
alias: torch
group: ml
config: config.py
depends: [python, numpy, onnx]
dockerfile: Dockerfile
test: test.py
```

This would be equivalent to having it encoded into the Dockerfile like above.

## JSON

Config files can also be provided in JSON format (normally called `config.json`).  The JSON and YAML configs typically get used when defining meta-containers that may not even have their own Dockerfiles, but exist solely as combinations of other packages - like [`l4t-pytorch`](/packages/l4t/l4t-pytorch) does:

```json
{
    "l4t-pytorch": {
        "group": "ml",
        "depends": ["pytorch", "torchvision", "torchaudio", "torch2trt", "opencv", "pycuda"]
    }
}
```

You can define multiple packages/containers per config file, like how [`l4t-tensorflow`](/packages/l4t/l4t-tensorflow) has versions for both TF1/TF2:

```json
{
    "l4t-tensorflow:tf1": {
        "group": "ml",
        "depends": ["tensorflow", "opencv", "pycuda"]
    },
    
    "l4t-tensorflow:tf2": {
        "group": "ml",
        "depends": ["tensorflow2", "opencv", "pycuda"]
    }
}
```

## Python

Python configuration scripts (normally called `config.py`) are the most expressive and get executed at the start of a build, and can dynamically set build parameters based on your environment and version of JetPack/L4T.  They have a global `package` dict added to their scope by the build system, which is used to configure the package:

```python
from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES

if L4T_VERSION.major >= 36: 
    MY_PACKAGE_VERSION = 'v6.0'  # on JetPack 6
elif L4T_VERSION.major == 35:
    MY_PACKAGE_VERSION = 'v5.0'  # on JetPack 5
else:                        
    MY_PACKAGE_VERSION = 'v4.0'  # on JetPack 4

package['build_args'] = {
    'MY_PACKAGE_VERSION': MY_PACKAGE_VERSION,
    'CUDA_ARCHITECTURES': ';'.join(CUDA_ARCHITECTURES),
}
```

This example sets build args in a Dockerfile, based on the version of JetPack/L4T that's running and the GPU architectures to compile for.  Typically the package's static settings remain in the Dockerfile header for the best visibility, while `config.py` sets the dynamic ones.


The [`jetson_containers`](/jetson_containers) module exposes these [system variables](/jetson_containers/l4t_version.py) that you can import and parameterize Dockerfiles off of:

| Name                 |                                       Type                                      | Description                                             |
|----------------------|:-------------------------------------------------------------------------------:|---------------------------------------------------------|
| `L4T_VERSION`        | [`packaging.version.Version`](https://packaging.pypa.io/en/latest/version.html) | version of L4T from `/etc/nv_tegra_release`             |
| `JETPACK_VERSION`    | [`packaging.version.Version`](https://packaging.pypa.io/en/latest/version.html) | version of JetPack corresponding to L4T version         |
| `PYTHON_VERSION`     | [`packaging.version.Version`](https://packaging.pypa.io/en/latest/version.html) | version of Python (`3.6` or `3.8`)                      |
| `CUDA_VERSION`       | [`packaging.version.Version`](https://packaging.pypa.io/en/latest/version.html) | version of CUDA (under `/usr/local/cuda`)               |
| `CUDA_ARCHITECTURES` |                                   `list[int]`                                   | NVCC GPU architectures for codegen (e.g. `[72,87,101]`) |
| `SYSTEM_ARCH`        |                                      `str`                                      | `aarch64` or `x86_64`                                   |
| `LSB_RELEASE`        |                                      `str`                                      | `18.04` or `20.04`                                      |
| `LSB_CODENAME`       |                                      `str`                                      | `bionic` or `focal`                                     |

Of course, it being Python, you can perform basically any other system queries/configuration you want using Python's built-in libraries, including manipulating files used by the build context, ect.
