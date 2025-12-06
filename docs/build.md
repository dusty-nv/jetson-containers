# Building Containers

The [`jetson-containers build`](/jetson-containers) command is a proxy launcher for [`jetson_containers/build.py`](/jetson_containers/build.py).  It can be run from any working directory after you clone the repo and run the installer from the [System Setup](/docs/setup.md) (you should also probably mount additional storage when building containers)

To list the packages available to build for your version of JetPack/L4T, you can use `--list-packages` and `--show-packages`:

```bash
$ jetson-containers build --list-packages       # list all packages
$ jetson-containers build --show-packages       # show all package metadata
$ jetson-containers build --show-packages ros*  # show all the ros packages
```

To build a container that includes one or more packages:

```bash
$ jetson-containers build pytorch               # build a container with just PyTorch
$ jetson-containers build pytorch jupyterlab    # build container with PyTorch and JupyterLab
```

The builder will chain together the Dockerfiles of each of packages specified, and use the result of one build stage as the base image of the next.  The initial base image defaults to one built for your version of JetPack, but can be changed by specifying the [`--base` argument](#changing-the-base-image).

The docker commands that get run during the build are printed and also saved to shell scripts under the `jetson-containers/logs` directory.  To see just the commands it would have run without actually running them, use the `--simulate` flag.

## Container Names

By default, the name of the container will be based on the package you chose to build. However, you can name it with the `--name` argument:

```bash
jetson-containers build --name=my_container pytorch jupyterlab
```

If you omit a tag from your name, then a default one will be assigned (based on the tag prefix/postfix of the package - typically `$L4T_VERSION`)

You can also build your container under a namespace ending in `/` (for example `--name=my_builds/` would result in an image like `my_builds/pytorch:r36.2.0`).  Using namespaces can help manage/organize your images if you are building a lot of them.

## Multiple Containers

The `--multiple` flag builds separate containers for each of the packages specified:

```bash
$ jetson-containers build --multiple pytorch tensorflow2   # built a pytorch container and a tensorflow2 container
$ jetson-containers build --multiple ros*                  # build all ROS containers
```

If you wish to continue building other containers if one fails, use the `--skip-errors` flag (this only applies to building multiple containers)

## Changing the Base Image

By default, the base container image used at the start of the build chain will be [`l4t-base`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-base) on JetPack 4, [`l4t-jetpack`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack) on JetPack 5, [`ubuntu:22.04`](https://hub.docker.com/_/ubuntu/tags?page=&page_size=&ordering=&name=22.04) on JetPack 6, [`ubuntu:24.04`](https://hub.docker.com/_/ubuntu/tags?page=&page_size=&ordering=&name=24.04) on JetPack 7.  However, if you want to add packages to a container that you already have, you can specify your own base image:

```bash
jetson-containers build --base=my_container:latest --name=my_container:pytorch pytorch  # add pytorch to your container
```

> [!NOTE]
> On JetPack 4/5, it's assumed that base images already have the JetPack components available/installed (CUDA Toolkit, cuDNN, TensorRT).
>
> On JetPack 6 (L4T r36.x) and JetPack 7 (L4T r38.x), the CUDA components will automatically be installed on top of your base image if required.
> > You can specify the CUDA version by using `CUDA_VERSION` environment variable, otherwise it will pick the CUDA version that was made available to the L4T version.

## Changing Versions

Many packages are versioned, and define subpackages like `pytorch:2.5`, `pytorch:2.6`, ect in the package's [configuration](/docs/packages.md#python).  For some core packages, you can control the default version of these with environment variables like `CUDA_VERSION`, `PYTORCH_VERSION`, `PYTHON_VERSION`, and `LSB_RELEASE` for the Ubuntu distro.  Other packages referring to these will then use your desired versions instead of the previous ones:

```bash
LSB_RELEASE=24.04 CUDA_VERSION=13.1 jetson-containers build --name=cu130 pytorch  # build PyTorch for Ubuntu 24.04 + CUDA 13.1
```

The dependencies are also able to specify with [`requires`](/docs/packages.md) which versions of L4T, CUDA, and Python they need, so changing the CUDA version has cascading effects downstream and will also change the default version of cuDNN, TensorRT, and PyTorch (similar to how changing the PyTorch version also changes the default version of torchvision and torchaudio).  The reverse also occurs in the other direction, for example changing the TensorRT version will change the default version of CUDA (unless you explicitly specify it otherwise).

| Package       | Environment Variable | Versions                                                                             |
|---------------|----------------------|-----------------------------------------------------------------------------------------|
| [`cuda`](/packages/cuda/cuda) | `CUDA_VERSION` | [`cuda/config.py`](/packages/cuda/cuda/config.py) |
| [`cudnn`](/packages/cuda/cudnn) | `CUDNN_VERSION` | [`cudnn/config.py`](/packages/cuda/cudnn/config.py) |
| [`tensorrt`](/packages/tensorrt) | `TENSORRT_VERSION` | [`tensorrt/config.py`](/packages/tensorrt/config.py) |
| [`python`](/packages/python) | `PYTHON_VERSION` | [`python/config.py`](/packages/python/config.py) |
| [`pytorch`](/packages/python) | `PYTORCH_VERSION` | [`pytorch/config.py`](/packages/pytorch/config.py) |
| [`ubuntu`](/jetson_containers/l4t_version.py) | `LSB_RELEASE` | [`l4t_version.py`](/jetson_containers/l4t_version.py) |
| [`cuda`](/jetson_containers/l4t_version.py) | `CUDA_ARCH` | [`l4t_version.py`](/jetson_containers/l4t_version.py) |

## What is CUDA_ARCH?

The following are the main CUDA-supported architectures and their differences:

---

### 🖥️ x86_64

**Definition:**  
64-bit x86 architecture used widely in modern desktops, laptops, and servers.

**Key Characteristics:**
- Common on Intel and AMD CPUs
- Compatible with most CUDA tools and GPUs
- Runs mainstream Linux and Windows OS

**Use Case:**  
General-purpose CUDA development and deployment on workstations or cloud servers.

---

### 📱 [aarch64-sbsa (Server Base System Architecture)](https://www.nvidia.com/en-us/data-center/grace-cpu/)

**Definition:**  
64-bit ARM architecture for **server-class hardware** following the SBSA specification.

**Key Characteristics:**
- Targeted at datacenter-Grace ARM CPUs (e.g., Ampere, AWS Graviton)
- Supports Linux OS with CUDA toolkits
- Designed for cloud-native and scalable environments

**Use Case:**  
Energy-efficient, high-performance computing in ARM-based server infrastructures.

---

### 🤖 [aarch64-jetson or cuda-tegra](https://developer.nvidia.com/embedded/jetpack)

**Definition:**  
64-bit ARM architecture specifically for **NVIDIA Jetson embedded systems**.

**Key Characteristics:**
- Designed for Jetson boards (Nano, Xavier, Orin, etc.)
- Built-in GPU for CUDA acceleration
- Runs Ubuntu-based Linux for Tegra (L4T)

**Use Case:**  
Robotics, edge AI, computer vision, and IoT solutions using CUDA at the edge.

---

## 📊 Summary Table

| Architecture       | CPU Type         | Target Device                  | CUDA Support | Use Case                          |
|--------------------|------------------|--------------------------------|--------------|------------------------------------|
| **x86_64**         | Intel / AMD      | PCs, Laptops, Servers          | ✅            | General CUDA development           |
| **aarch64-sbsa**     | ARM (Server)     | Datacenter-Grace ARM servers   | ✅            | Efficient cloud compute            |
| **aarch64-jetson / cuda-tegra** | ARM (Embedded)   | NVIDIA Jetson Boards           | ✅            | Edge AI and embedded applications  |

Using these together, you can rebuild the container stack for the specific version combination that you want:

```bash
LSB_RELEASE=24.04 CUDA_VERSION=13.1 PYTHON_VERSION=3.12 PYTORCH_VERSION=2.10 jetson-containers build vllm
```


The available versions are defined in the package's configuration scripts (some you can simply add new releases to by referring to the new git tag/branch, others need pointed to specific release downloads).  Not all combinations are compatible, as the versioned packages frequently define the minimum version of others in the build tree they rely on (for example, TensorRT 10 requires CUDA 12.4).

For packages that provide different versions but don't have their own environment variable defined, you can specify the desired version of them that your container depends on in the Dockerfile header under [`depends`](/docs/packages.md), and it will override the default and build using them instead (e.g. by using `depends: onnxruntime:1.17` instead of `depends: onnxruntime`).  Like above, these versions will also cascade across the build.

> [!NOTE]
> > On JetPack 7 (L4T r38.x), if you don't have `CUDA_VERSION` specified, it will pick the CUDA version that was packaged with that version of L4T in the JetPack.
> - L4T r38.1.x (JetPack 7.0+) --> CUDA `13.0`
> On JetPack 6 (L4T r36.x), if you don't have `CUDA_VERSION` specified, it will pick the CUDA version that was packaged with that version of L4T in the JetPack.
> - L4T r36.4.x (JetPack 6.1+) --> CUDA `12.6`
> - L4T r36.2 (JetPack 6.0 DP) --> CUDA `12.2`
> - L4T r36.3 (JetPack 6.0 GA) --> CUDA `12.4`

## 24.04 Containers

Here is a list of containers currently built for Ubuntu 24.04 with the following environment:

* `LSB_RELEASE=24.04 L4T_VERSION=36.4.4`
* `CUDA_VERSION=13.1 CUDNN_VERSION=9.17`
* `PYTHON_VERSION=3.12 PYTORCH_VERSION=2.8`

| Repo           | Version   | Image                                             |  Size (GB)  | Timestamp   |
|:--------------:|:---------:|:-------------------------------------------------:|:-----------:|:-----------:|
| sglang         | 0.4.4     | `dustynv/sglang:0.4.4-r36.4.4-cu128-24.04`        |     5.4     | 2025-03-03  |
| bitsandbytes   | 0.45.4    | `dustynv/bitsandbytes:0.45.4-r36.4.4-cu128-24.04` |     4.4     | 2025-03-03  |
| flashinfer     | 0.2.3     | `dustynv/flashinfer:0.2.3-r36.4.4-cu128-24.04`    |     3.9     | 2025-03-03  |
| jupyterlab     | latest    | `dustynv/jupyterlab:r36.4.4-cu128-24.04`          |     5.1     | 2025-03-03  |
| kokoro-tts     | fastapi   | `dustynv/kokoro-tts:fastapi-r36.4.4-cu128-24.04`  |     4.8     | 2025-03-03  |
| llama_cpp      | 0.3.7     | `dustynv/llama_cpp:0.3.7-r36.4.4-cu128-24.04`     |     3.2     | 2025-03-03  |
| piper-tts      | latest    | `dustynv/piper-tts:r36.4.4-cu128-24.04`           |     5.6     | 2025-03-03  |
| faster-whisper | latest    | `dustynv/faster-whisper:r36.4.4-cu128-24.04`      |     5.3     | 2025-03-03  |
| onnxruntime    | 1.22      | `dustynv/onnxruntime:1.22-r36.4.4-cu128-24.04`    |     5.2     | 2025-03-03  |
| cupy           | latest    | `dustynv/cupy:r36.4.4-cu128-24.04`                |     2.3     | 2025-03-03  |
| pycuda         | latest    | `dustynv/pycuda:r36.4.4-cu128-24.04`              |     2.3     | 2025-03-03  |
| cuda-python    | latest    | `dustynv/cuda-python:r36.4.4-cu128-24.04`         |     2.3     | 2025-03-03  |
| speaches       | latest    | `dustynv/speaches:r36.4.4-cu128-24.04`            |     6.6     | 2025-03-03  |
| jax            | 0.5.2     | `dustynv/jax:0.5.2-r36.4.4-cu128-24.04`           |     3.6     | 2025-03-03  |
| ros            | jazzy     | `dustynv/ros:jazzy-ros-base-r36.4.4-cu128-24.04`  |     5.1     | 2025-03-03  |
| ros            | jazzy     | `dustynv/ros:jazzy-desktop-r36.4.4-cu128-24.04`   |     5.9     | 2025-03-03  |
| torch2trt      | latest    | `dustynv/torch2trt:r36.4.4-cu128-24.04`           |     5.2     | 2025-03-03  |
| opencv         | 4.11.0    | `dustynv/opencv:4.11.0-r36.4.4-cu128-24.04`       |     3.3     | 2025-03-02  |
| vllm           | 0.7.4     | `dustynv/vllm:0.7.4-r36.4.4-cu128-24.04`          |     5.3     | 2025-02-28  |
| openvla        | cp312     | `dustynv/openvla:r36.4.3-cu128-cp312-24.04`       |     6.7     | 2025-02-27  |
| torchao        | 0.11.0    | `dustynv/torchao:0.11.0-r36.4.4-cu128-24.04`      |     3.5     | 2025-02-26  |
| torchaudio     | 2.8.0     | `dustynv/torchaudio:2.8.0-r36.4.4-cu128-24.04`    |     3.5     | 2025-02-26  |
| torchvision    | 0.23.0    | `dustynv/torchvision:0.23.0-r36.4.4-cu128-24.04`  |     3.5     | 2025-02-26  |
| pytorch        | 2.8       | `dustynv/pytorch:2.8-r36.4.4-cu128-24.04`         |     3.5     | 2025-02-26  |

You can build or run these from JetPack 6.1+ like the following:

* Build: `LSB_RELEASE=24.04 jetson-containers build pytorch:2.8` 
* Run:  `jetson-containers run dustynv/pytorch:2.8-r36.4.4-cu128-24.04`

## Pip Server

Being able to change the versions of CUDA, Python, ect in the build tree will cause all the dependant packages like PyTorch to need rebuilt, and this can be time-consuming to recompile everything from scratch.  In order to cache and re-use the Python wheels compiled during the container builds, there's a [pip server](http://pypi.jetson-ai-lab.io) running that wheels are uploaded to.  The containers automatically use this pip server via the `PIP_INDEX_URL` environment variable that gets set in the base containers, and there are indexes created for different versions of JetPack/CUDA.

Then when the containers are built, it will first attempt to install the wheel from the server, and if it's found it builds it from source.  There is another similar server run for caching [tarball releases](http://pypi.jetson-ai-lab.io:8000) for projects that install C/C++ binaries and headers or other general files outside of pip.  See this [forum post](https://forums.developer.nvidia.com/t/jetson-ai-lab-ml-devops-containers-core-inferencing/288235/3) for more information about the caching infrastructure, and the ability to use the pip server outside of just containers.

### Local `PIP` and `APT` servers

You can start the local `PYPI`/`APT` servers on one Jetson node with a fallback set to `jetson-ai-lab.io`. The local builded `pip` wheels and `tarballs` will be uploaded from the container during build time to local `PYPI`/`APT` instances only.

To start local `pypi`/`apt` servers:
```bash
docker compose -f jetson_containers/packages/net/docker-compose.develop.yaml up --detach
```

To stop local `pypi`/`apt` servers:
```bash
docker compose -f jetson_containers/packages/net/docker-compose.develop.yaml stop
```

### How to activate the environment?

```bash
git clone --recursive https://github.com/dusty-nv/jetson-containers.git
sudo bash jetson-containers/install.sh
cd jetson-containers
# modify here your .env for example: 
INDEX_HOST=jetson-ai-lab.io
# activate it
source .env
# Then you can build with your variables
jetson-containers build torch
```

## Building External Packages

Let's say that you have a project that you want to build a container for - if you define a [custom package](/docs/packages.md) (i.e. by creating a Dockerfile with a YAML header), you can point `build.sh` to out-of-tree locations using the `--package-dirs` option:

```bash
jetson-containers build --package-dirs=/path/to/your/package your_package_name
```

You can also add jetson-containers as a git submodule to your project and build it that way (see [jetson-inference](https://github.com/dusty-nv/jetson-inference) as an example of this)

## Tests

By default, tests are run during container builds for the packages that provide test scripts.  After each stage of the build chain is complete, that package will be tested.  After the build is complete, the final container will be tested against all of the packages again.  This is to assure that package versions aren't inadvertantly changed/overwritten/uninstalled at some point during the build chain (typically, the GPU-accelerated version of the package being supplanted by the one that some other package installs from pip/apt/ect)

You can skip these tests however with the `--skip-tests` argument, which is a comma/colon-separated list of package names to skip their tests.  Or `all` or `*` will skip all tests, while `intermediate` will only run tests at the end (not during).

``` bash
$ jetson-containers build --skip-tests=numpy,onnx pytorch    # skip testing the numpy and onnx packages when building pytorch
$ jetson-containers build --skip-tests=all pytorch           # skip all tests
$ jetson-containers build --skip-tests=intermediate pytorch  # only run tests at the end of the container build
```

These flags tend to get used more during development - normally it's good to thoroughly test the build stack, as to not run into cryptic issues with packages later on.

## Running Containers

To launch containers that you've built or pulled, see the [Running Containers](/docs/run.md) documentation or the package's readme page.

## Build cache invalidation

Add `--build-flags="--no-cache"`.

```bash
$ jetson-containers build --build-flags="--no-cache" lerobot
```

## Troubleshooting

When building a container, you may hit this GitHub API rate limit especially if you are working in an office environment or similar where your outward-facing IP address is shared with other developers/instances.

> `ADD failed: failed to GET https://api.github.com/repos/abetlen/llama-cpp-python/git/refs/heads/main with status 403 Forbidden: {"message":"API rate limit exceeded for 216.228.112.22...`

If that is the case, use `--no-github-api` option when running `build.sh`

```
$ jetson-containers build --no-github-api --skip-test=all text-generation-webui
```

The option `--no-github-api` is to remove a line like below from `Dockerfile` that was added to force rebuild on new git commits.

```
ADD https://api.github.com/repos/${LLAMA_CPP_PYTHON_REPO}/git/refs/heads/${LLAMA_CPP_PYTHON_BRANCH} /tmp/llama_cpp_python_version.json
```

You will find `Dockerfile.minus-github-api` file newly created in each package directory if the Dockerfile contains such line, and that's what used for building.  The `Dockerfile.minus-github-api` is temporary (and is listed in `.gitignore`), so always edit the original `Dockerfile` when needed.  If you want to remove all such temporary files, you can execute the following command.

> `find . -type f -name *.minus-github-api -delete`
