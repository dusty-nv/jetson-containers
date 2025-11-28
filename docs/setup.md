# System Setup

Install the latest version of JetPack 4 on Nano/TX1/TX2, JetPack 5 on Xavier, or JetPack 6 on Orin.  The following versions are supported:

* JetPack 4.6.1+ (>= L4T R32.7.1)
* JetPack 5.1+  (>= L4T R35.2.1)
* JetPack 6.0 DP (L4T R36.2.0)
* JetPack 6.2 DP (L4T R36.4.4)
> [!NOTE]  
> <sup>- Building on/for x86 platforms isn't supported at this time (one can typically install/run packages the upstream way there)</sup><br>
> <sup>- The below steps are optional for [pulling/running](/docs/run.md) existing container images from registry, but recommended for building containers locally.</sup>

## Clone the Repo

This will download and install the jetson-containers utilities:

```bash
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers
bash install.sh #run with sudo if needed
```

The installer script will prompt you for your sudo password, and will setup some Python [requirements](/requirements.txt) and add tools like [`autotag`](/docs/run.md#autotag) the `$PATH` by linking them under `/usr/local/bin` (if you move your jetson-containers repo, run this step again)

If you are only running containers and already have enough disk space on your root drive to download them, you may be able to skip the rest of the steps below, but they are recommended best-practices and should be followed when building your own containers.

## Docker Default Runtime

If you're going to be building containers, you need to set Docker's `default-runtime` to `nvidia`, so that the NVCC compiler and GPU are available during `docker build` operations.  Add `"default-runtime": "nvidia"` to your `/etc/docker/daemon.json` configuration file before attempting to build the containers:

``` json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "default-runtime": "nvidia"
}
```

Then restart the Docker service, or reboot your system before proceeding:

```bash
$ sudo systemctl restart docker
```

You can then confirm the changes by looking under `docker info`

```bash
$ sudo docker info | grep 'Default Runtime'
Default Runtime: nvidia
```

## Relocating Docker Data Root

Containers can take up a lot of disk space.  If you have external storage available, it's advised to relocate your Docker container cache to the larger drive (NVME is preferred if possible).  If it's not already, get your drive formatted as ext4 and so that it's mounted at boot (i.e. it should be in `/etc/fstab`).  If it's not automatically mounted at boot before the Docker daemon starts, the directory won't exist for Docker to use.

Copy the existing Docker cache from `/var/lib/docker` to a directory on your drive of choice (in this case, `/mnt/docker`):

```bash
$ sudo cp -r /var/lib/docker /mnt/docker
```

Then add your directory as `"data-root"` in `/etc/docker/daemon.json`:

``` json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "default-runtime": "nvidia",
    "data-root": "/mnt/docker"
}
```

Then restart the Docker service, or reboot your system before proceeding:

```bash
$ sudo systemctl restart docker
```

You can then confirm the changes by looking under `docker info`

```bash
$ sudo docker info | grep 'Docker Root Dir'
Docker Root Dir: /mnt/docker
...
Default Runtime: nvidia
```

That directory will also now have had it's permissions changed to root-access only by the Docker daemon.

## Mounting Swap

If you're building containers or working with large models, it's advisable to mount SWAP (typically correlated with the amount of memory in the board).  Run these commands to disable ZRAM and create a swap file:

``` bash
sudo systemctl disable nvzramconfig
sudo fallocate -l 16G /mnt/16GB.swap
sudo mkswap /mnt/16GB.swap
sudo swapon /mnt/16GB.swap
```
> If you have NVME storage available, it's preferred to allocate the swap file on NVME.

Then add the following line to the end of `/etc/fstab` to make the change persistent:

``` bash
/mnt/16GB.swap  none  swap  sw 0  0
```

## Disabling the Desktop GUI

If you're running low on memory, you may want to try disabling the Ubuntu desktop GUI.  This will free up extra memory that the window manager and desktop uses (around ~800MB for Unity/GNOME or ~250MB for LXDE)  

You can disable the desktop temporarily, run commands in the console, and then re-start the desktop when desired: 

``` bash
$ sudo init 3     # stop the desktop
# log your user back into the console (Ctrl+Alt+F1, F2, ect)
$ sudo init 5     # restart the desktop
```

If you wish to make this persistent across reboots, you can use the follow commands to change the boot-up behavior:

``` bash
$ sudo systemctl set-default multi-user.target     # disable desktop on boot
$ sudo systemctl set-default graphical.target      # enable desktop on boot
```

## Adding User to Docker Group

Seeing as Ubuntu users aren't by default in the `docker` group, they need to run docker commands with `sudo` (the build tools automatically do this when needed).  Hence you could be periodically asked for your sudo password during builds.  

Instead, you can add your user to the docker group like below:

```bash
sudo usermod -aG docker $USER
```

Then close/restart your terminal (or logout) and you should be able to run docker commands (like `docker info`) without needing sudo.

## Setting the Power Mode

Depending on the power supply source you have available for your Jetson (i.e. wall power or battery), you may wish to put your Jetson in maximum power mode (MAX-N) to attain the highest performance available from your Jetson device.  You can do this with the [`nvpmodel`](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#power-mode-controls) command-line tool, or from the Ubuntu desktop via the [nvpmodel GUI widget](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#nvpmodel-gui) (or by using [`jtop`](https://github.com/rbonghi/jetson_stats) from jetson-stats)

```bash
# check the current power mode
$ sudo nvpmodel -q
NV Power Mode: MODE_30W
2

# set it to mode 0 (typically the highest)
$ sudo nvpmodel -m 0

# reboot if necessary, and confirm the changes
$ sudo nvpmodel -q
NV Power Mode: MAXN
0
```

See [here](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#supported-modes-and-power-efficiency) for a table of the power modes available for the different Jetson devices, and for documentation on the [`nvpmodel`](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#power-mode-controls) tool.

