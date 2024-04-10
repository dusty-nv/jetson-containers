# homeassistant-core

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

## Home Assistant Core

<p align="center"><img src="ha_onboarding.png" title="Home Assistant Core" alt="Home Assistant Core onboarding screen" /></p>

This image is based on `Home Assistant Core`:

| | `HA OS` | `Container` | `Core` | `Supervised` |
|---|---|---|---|---|
| Automations | ✅ | ✅ | ✅ | ✅ |
| Dashboards | ✅ | ✅ | ✅ | ✅ |
| Integrations | ✅ | ✅ | ✅ | ✅ |
| Blueprints | ✅ | ✅ | ✅ | ✅ |
| Uses container | ✅ | ✅ | ❌ | ✅ |
| Supervisor | ✅ | ❌ | ❌ | ✅ |
| Add-ons | ✅ | ❌ | ❌ | ✅ |
| Backups | ✅ | ✅ | ✅ | ✅ |
| Managed Restore | ✅ | ❌ | ❌ | ✅ |
| Managed OS | ✅ | ❌ | ❌ | ❌ |

### Onboarding

The Webui can be found at http://your-ip:8123. (replace with the hostname or IP of the system) Follow the wizard to set up Home Assistant. 

Feel free to follow detailed instructions here: https://www.home-assistant.io/getting-started/onboarding/

### Configuration files location

You can specify where you want to store your Home Assistant Core configuration by attaching a `volume`. Make sure that you keep the `:/config` part:

```sh
-v /PATH_TO_YOUR_CONFIG:/config
```

### Configure auto-discover

`Home Assistant` can discover and automatically configure `zeroconf`/`mDNS` and `UPnP` devices on your network. In order for this to work you must create the container with `--net=host`.

#### Docker Cli:
```sh
--net=host
```

#### Docker Compose:
```sh
    network_mode: host
```

### Accessing Bluetooth Device

In order to provide `HA` with access to the host's `Bluetooth` device, Home Assistant uses `BlueZ` on the `host` - add the capabilities `NET_ADMIN` and `NET_RAW` to the container, and map `dbus` as a `volume` as shown in the below examples to enable Bluetooth support:

#### Docker Cli:
```sh
--cap-add=NET_ADMIN --cap-add=NET_RAW -v /var/run/dbus:/var/run/dbus:ro
```

#### Docker Compose:
```sh
    cap_add:
      - NET_ADMIN
      - NET_RAW
    volumes:
      - /var/run/dbus:/var/run/dbus:ro
```

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`homeassistant-core:latest`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `homeassistant-core` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`homeassistant-base`](/packages/smart-home/homeassistant-base) [`ffmpeg`](/packages/ffmpeg) [`python:3.12`](/packages/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The `homeassistant-core` wheel that's build is saved in `/usr/src/homeassistant` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/homeassistant-core:2024.4.2-r35.4.1`](https://hub.docker.com/r/dustynv/homeassistant-core/tags) | `2024-04-09` | `arm64` | `6.0GB` |
| &nbsp;&nbsp;[`dustynv/homeassistant-core:2024.4.2-r36.2.0`](https://hub.docker.com/r/dustynv/homeassistant-core/tags) | `2024-04-09` | `arm64` | `1.4GB` |
| &nbsp;&nbsp;[`dustynv/homeassistant-core:r35.4.1`](https://hub.docker.com/r/dustynv/homeassistant-core/tags) | `2024-04-10` | `arm64` | `6.0GB` |
| &nbsp;&nbsp;[`dustynv/homeassistant-core:r36.2.0`](https://hub.docker.com/r/dustynv/homeassistant-core/tags) | `2024-04-10` | `arm64` | `1.4GB` |

> <sub>Container images are compatible with other minor versions of JetPack/L4T:</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R32.7 containers can run on other versions of L4T R32.7 (JetPack 4.6+)</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R35.x containers can run on other versions of L4T R35.x (JetPack 5.1+)</sub><br>
</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use the [`run.sh`](/docs/run.md)/[`autotag`](/docs/run.md#autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag homeassistant-core)

# or explicitly specify one of the container images above
./run.sh dustynv/homeassistant-core:r35.4.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/homeassistant-core:r35.4.1
```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag homeassistant-core)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag homeassistant-core) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh homeassistant-core
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
