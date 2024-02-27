# docker

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

This container has the Docker CLI tools installed in it, and can perform Docker operations (such as starting/stopping containers) through the host's Docker daemon.  Access to the host's Docker daemon should be mounted with `--volume /var/run/docker.sock:/var/run/docker.sock` (which is automatically done by [`jetson-containers/run.sh`](/run.sh)).  Then it will share all the same container images and instances that are available on the host.

This is not technically Docker-in-Docker, as the container is not running its own Docker daemon (but rather sharing the host's).  For more info, see [Jérôme Petazzoni's excellent blog post on the subject](https://jpetazzo.github.io/2015/09/03/do-not-use-docker-in-docker-for-ci/), which outlines the pro's and con's and common pitfalls of these approaches.  In particular, mounting the Docker socket as mentioned above allieviates many of these issues and does not require the `--privileged` flag.

This approach works with `--runtime nvidia` and access to the GPU.  Note that if you're starting a container within this container and trying to mount volumes, the paths are referenced from the host (see https://stackoverflow.com/a/31381323)
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`docker`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=32.6` |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use the [`run.sh`](/docs/run.md)/[`autotag`](/docs/run.md#autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag docker)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host docker:36.2.0

```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag docker)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag docker) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh docker
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
