# Building Containers

[`build.sh`](/build.sh) is a proxy launcher script for [`/jetson_containers/build.py`].  It can be run from any working directory (in the examples below, that's assumed to be your jetson-containers repo).  Make sure you do the [System Setup](/docs/setup.md) first.

To list the packages available to build for your version of JetPack/L4T, you can use `--list-packages` and `--show-packages`:

```bash
$ ./build.sh --list-packages      # list all packages
$ ./build.sh --show-packages      # show all package metadata
$ ./build.sh --show-packages ros* # show all the ros packages
```

To build a container that includes one or more packages:

```bash
$ ./build.sh pytorch              # build a container with just PyTorch
$ ./build.sh pytorch jupyterlab   # build container with PyTorch and JupyterLab
```

The builder will chain together the Dockerfiles of each of packages specified, and use the result of one build stage as the base image of the next.  The initial base image defaults to `l4t-base`/`l4t-jetpack` (for JetPack 4 and JetPack 5, respectively) but can be changed by specifying the `--base-image` argument.

The docker commands that get run during the build are printed and also saved to shell scripts under the `jetson-containers/logs` directory.  To see just the commands it would have run without actually running them, use the `--simulate` flag.

## Container Names

By default, the name of the container will be based on the package you chose to build. However, you can name it with the `--name` argument:

```bash
$ ./build.sh --name=my_container pytorch jupyterlab
```

If you omit a tag from your name, then a default one will be assigned (based on the tag prefix/postfix of the package - typically `$L4T_VERSION`)

You can also build your container under a namespace (ending in `/`), for example `--name=my_builds/` would result in an image like `my_builds/pytorch:r35.2.1` (doing that can help manage/organize your images if you have a lot)

## Multiple Containers

The `--multiple` flag builds separate containers for each of the packages specified:

```bash
$ ./build.sh --multiple pytorch tensorflow2   # built a pytorch container and a tensorflow2 container
$ ./build.sh --multiple ros*                  # build all ROS containers
```

If you wish to continue building other containers if one fails, use the `--skip-errors` flag (this only applies to building multiple containers)

## Changing the Base Image

By default, the base container image used at the start of the build chain will be [`l4t-base`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-base) (on JetPack 4) or [`l4t-jetpack`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack) (on JetPack 5).  However, if you want to add packages to a container that you already have, you can specify your own base image with the `--base` flag:

```bash
$ ./build.sh --base=my_container:latest --name=my_container:pytorch pytorch  # add pytorch to your existing container
```

> note:  it's assumed that base images have the JetPack components available/installed (i.e. CUDA Toolkit, cuDNN, TensorRT)

## Building External Packages

Let's say that you have a project that you want to build a container for - if you define a [custom package](/docs/packages.md) (i.e. by creating a Dockerfile with a YAML header), you can point `build.sh` to out-of-tree locations using the `--package-dirs` option:

```bash
$ ./build.sh --package-dirs=/path/to/your/package your_package_name
```

You can also add jetson-containers as a git submodule to your project and build it that way - see [jetson-inference](https://github.com/dusty-nv/jetson-inference) as an example of this.

## Tests

By default, tests are run during container builds for the packages that provide test scripts.  After each stage of the build chain is complete, that package will be tested.  After the build is complete, the final container will be tested against all of the packages again.  This is to assure that package versions aren't inadvertantly changed/overwritten/uninstalled at some point during the build chain (typically, the GPU-accelerated version of the package being supplanted by the one that some other package installs from pip/apt/ect)

You can skip these tests however with the `--skip-tests` argument, which is a comma/colon-separated list of package names to skip their tests.  Or `all` or `*` will skip all tests, while `intermediate` will only run tests at the end (not during).

``` bash
$ ./build.sh --skip-tests=numpy,onnx pytorch    # skip testing the numpy and onnx packages when building pytorch
$ ./build.sh --skip-tests=all pytorch           # skip all tests
$ ./build.sh --skip-tests=intermediate pytorch  # only run tests at the end of the container build
```

These flags tend to get used more during development - normally it's good to thoroughly test the build stack, as to not run into cryptic issues with packages later on.

## Running Containers

To launch containers that you've built or pulled, see the [Running Containers](/docs/run.md) page or the package's readme.
