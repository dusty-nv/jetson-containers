# System Setup

Install the latest version of JetPack 4 if you are on Nano/TX1/TX2, or JetPack 5 if you are on Xavier/Orin.  This repo tests on the following versions of JetPack:

* JetPack 4.6.1+ (>= L4T R32.7.1)
* JetPack 5.1+  (>= L4T R35.2.1)

## Clone the Repo

```bash
$ sudo apt-get update && sudo apt-get install git python3-pip
$ git clone https://github.com/dusty-nv/jetson-containers
$ cd jetson-containers
$ pip3 install -r requirements.txt
```

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

You can then confirm it by looking under `docker info`

```bash
$ sudo docker info | grep 'Default Runtime'
Default Runtime: nvidia
```

## Relocating Docker Data Root

Containers can take up a lot of disk space.  If you have external storage available, it's advised to relocate your Docker container cache to the larger drive (NVME is preferred if possible).  If it's not already, get your drive formatted as ext4 and so that it's mounted at boot (i.e. it should be in `/etc/fstab`).  If it's not automatically mounted at boot before the Docker daemon starts, the directory won't exist for Docker to use.

Copy the existing Docker cache from `/var/lib/docker` to a directory on your drive of choice (in this case, `/mnt/docker`):

```bash
sudo cp -r /var/lib/docker /mnt/docker
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

You can then confirm it by looking under `docker info`

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

## sudo NOPASSWD

Seeing as Ubuntu users aren't by default in the `docker` group, they need to run docker commands with `sudo`, as do the build tools.  Hence you could be randomly asked for your sudo password during builds.  To disable sudo from repeatedly asking you for your password, run `sudo visudo` and add this to the end:

```
your_username ALL=(ALL) NOPASSWD: ALL
```

Or to not ask for sudo password only when a docker command is being used:

```
your_username ALL=(ALL) NOPASSWD: /usr/bin/docker
```
