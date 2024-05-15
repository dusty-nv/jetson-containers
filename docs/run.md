# Running Containers

Let's say that you found a container image from the [Package List](/packages) or [DockerHub](https://hub.docker.com/u/dustynv), or [built your own container](/docs/build.md) - the normal way to run an interactive Docker container on your Jetson would be using [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) like this:

``` bash
$ sudo docker run --runtime nvidia -it --rm --network=host CONTAINER:TAG
```

That's actually a rather minimal command, and doesn't have support for displays or other devices, and it doesn't mount the model/data cache ([`/data`](/data)). Once you add everything in, it can get to be a lot to specify by hand.  Hence, we have some helpers that provide shortcuts.

The [`jetson-containers run`](/run.sh) launcher can be run from any directory and forwards its command-line to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), with some added defaults - including the above flags, mounting the `/data` cache, and mounting various devices for display, audio, and video (like V4L2 and CSI cameras)

``` bash
$ jetson-containers run CONTAINER:TAG                   # run with --runtime=nvidia, default mounts, ect
$ jetson-containers run CONTAINER:TAG my_app --abc xyz  # run a command (instead of interactive mode)
$ jetson-containers run --volume /path/on/host:/path/in/container CONTAINER:TAG  # mount a directory
```

The flags and arguments to [`jetson-containers run`](/run.sh) are the same as they are to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) - anything you specify will be passed along.

## `autotag`

To solve the issue of finding a container with package(s) that you want and that's compatible with your version of JetPack/L4T, there's the [`autotag`](/autotag) tool.  It locates a container image for you - either locally, pulled from a registry, or built from source:

``` bash
$ jetson-containers run $(autotag pytorch)   # find pytorch container to run for your version of JetPack/L4T
```

What's happening here with the `$(autotag xyz)` syntax, is that Bash command substitution expands the full container image name and forwards it to the `docker run` command.  For example, if you do `echo $(autotag pytorch)` it would print out something like `dustynv/pytorch:r35.2.1` (assuming that you don't already have the `pytorch` image locally).

You can of course use [`autotag`](/autotag) interspersed along with other command-line arguments to launch the container:

``` bash
$ jetson-containers run $(autotag pytorch) my_app --abc xyz  # run a command (instead of interactive mode)
$ jetson-containers run --volume /path/on/host:/path/in/container $(autotag pytorch)  # mount a directory
```

Or with using [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) directly:

``` bash
$ sudo docker run --runtime nvidia -it --rm --network=host $(./autotag pytorch)
```

This is the order in which [`autotag`](/autotag) searches for container images:

1. Local images (found under `docker images`)
2. Pulled from registry (by default [`hub.docker.com/u/dustynv`](https://hub.docker.com/u/dustynv))
3. Build it from source (it'll ask for confirmation first)

When searching for images, it knows to find containers that are compatible with your version of JetPack-L4T.  For example, if you're on JetPack 4.6.x (L4T R32.7.x), you can run images that were built for other versions of JetPack 4.6.  Or if you're on JetPack 5.1 (L4T R35), you can run images built for other versions of JetPack 5.1 (and likewise for JetPack 6.0 and newer)

## `jtop`

If you have installed [**jetson-stats**](https://github.com/rbonghi/jetson_stats) (or `jtop`) on your host, now a container with jetson-stats (`jtop`) installed can work inside the container by communicating with host server through a socket `/run/jtop.sock` (with `-v /run/jtop.sock:/run/jtop.sock` argument for `docker run`).

Check the [official documentation](https://rnext.it/jetson_stats/docker.html) for the detail.

Make sure you install the same version of jetson-stats (`jtop`) both on your host and in the container.