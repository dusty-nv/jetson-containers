# Building Containers

The [`jetson-containers build`](/jetson-containers) command is a proxy launcher for [`jetson_containers/build.py`](/jetson_containers/build.py).  It can be run from any working directory after you clone the repo and run the installer from the [System Setup](/docs/setup.md) (you should also probably mount additional storage when building containers, as they can get quite large)

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

The builder will chain together the Dockerfiles of each of packages specified, and use the result of one build stage as the base image of the next.  The initial base image defaults to `l4t-base`, `l4t-jetpack`, or `ubuntu:22.04` (for JetPack 4, 5, and 6, respectively) but can be changed by specifying the `--base-image` argument.

The docker commands that get run during the build are printed and also saved to shell scripts under the `jetson-containers/logs` directory.  To see just the commands it would have run without actually running them, use the `--simulate` flag.

## Container Names

By default, the name of the container will be based on the package you chose to build. However, you can name it with the `--name` argument:

```bash
$ jetson-containers build --name=my_container pytorch jupyterlab
```

If you omit a tag from your name, then a default one will be assigned (based on the tag prefix/postfix of the package - typically `$L4T_VERSION`)

You can also build your container under a namespace (ending in `/`), for example `--name=my_builds/` would result in an image like `my_builds/pytorch:r35.2.1` (doing that can help manage/organize your images if you have a lot)

## Multiple Containers

The `--multiple` flag builds separate containers for each of the packages specified:

```bash
$ jetson-containers build --multiple pytorch tensorflow2   # built a pytorch container and a tensorflow2 container
$ jetson-containers build --multiple ros*                  # build all ROS containers
```

If you wish to continue building other containers if one fails, use the `--skip-errors` flag (this only applies to building multiple containers)

## Changing the Base Image

By default, the base container image used at the start of the build chain will be [`l4t-base`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-base) on JetPack 4, [`l4t-jetpack`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack) on JetPack 5, and [`ubuntu:22.04`](https://hub.docker.com/_/ubuntu/tags?page=&page_size=&ordering=&name=22.04) on JetPack 6.  However, if you want to add packages to a container that you already have, you can specify your own base image:

```bash
$ jetson-containers build --base=my_container:latest --name=my_container:pytorch pytorch  # add pytorch to your container
```

> [!NOTE]  
> On JetPack 4/5, it's assumed that base images already have the JetPack components available/installed (CUDA Toolkit, cuDNN, TensorRT).  On JetPack 6, the CUDA components will automatically be installed on top of your base image if required.

## Building External Packages

Let's say that you have a project that you want to build a container for - if you define a [custom package](/docs/packages.md) (i.e. by creating a Dockerfile with a YAML header), you can point `build.sh` to out-of-tree locations using the `--package-dirs` option:

```bash
$ jetson-containers build --package-dirs=/path/to/your/package your_package_name
```

You can also add jetson-containers as a git submodule to your project and build it that way - see [jetson-inference](https://github.com/dusty-nv/jetson-inference) as an example of this.

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