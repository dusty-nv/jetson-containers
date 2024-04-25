## Home Assistant Core

<p align="center"><img src="ha_onboarding.png" title="Home Assistant Core" alt="Home Assistant Core onboarding screen" /></p>

This project was created by [Jetson AI Lab Research Group](https://www.jetson-ai-lab.com/research.html). The container image is based on `Home Assistant Core` and spans over below features:

| | `HA OS` | `Container` | `Core` | `Supervised` |
|---|---|---|---|---|
| Automations | ✅ | ✅ | ✅ | ✅ |
| Dashboards | ✅ | ✅ | ✅ | ✅ |
| Integrations | ✅ | ✅ | ✅ | ✅ |
| Blueprints | ✅ | ✅ | ✅ | ✅ |
| Uses container | ✅ | ✅ | ✅ | ✅ |
| Supervisor | ✅ | ❌ | ❌ | ✅ |
| Add-ons | ✅ | ❌ | ✅* | ✅ |
| Backups | ✅ | ✅ | ✅ | ✅ |
| Managed Restore | ✅ | ❌ | ❌ | ✅ |
| Managed OS | ✅ | ❌ | ❌ | ❌ |

> \* *Supports only manually installed `wyoming`–enabled Voice Assistant Add-ons from this repository*

### How to run using `docker-compose`

```yaml
name: home-assistant-jetson
version: "3.9"
services:
  homeassistant:
    image: dusty-nv/homeassistant-core:latest-r36.2.0-cu122-cp310
    restart: unless-stopped
    init: false
    privileged: true
    network_mode: host
    container_name: homeassistant
    hostname: homeassistant
    ports:
      - "8123:8123"
    volumes:
      - ha-config:/config
      - /etc/localtime:/etc/localtime:ro
      - /etc/timezone:/etc/timezone:ro
    stdin_open: true
    tty: true

volumes:
  ha-config:
```

### Onboarding

The UI can be found at http://your-ip:8123. *(replace with the `hostname` or `IP` of the system)*. Follow the wizard to set up Home Assistant. Feel free to follow [official instructions](https://www.home-assistant.io/getting-started/onboarding/).

### How to's

We encourage to look for help on [official Home Assistant documentation](https://www.home-assistant.io/docs/) and within the [HA Community Forums](https://community.home-assistant.io/) or on a [Jetson Research Group](https://www.jetson-ai-lab.com/research.html) [thread on NVIDIA forum](https://forums.developer.nvidia.com/t/jetson-ai-lab-home-assistant-integration/288225).

<details>
<summary><b>Configuration files location</b></summary>
<hr>

You can specify where you want to store your Home Assistant Core configuration by attaching a docker `volume`. Make sure that you keep the `:/config` part:

```sh
-v /PATH_TO_YOUR_CONFIG:/config
```
<hr>
<br>
</details>

<details>
<summary><b>Devices auto-discovery</b></summary>
<hr>

Home Assistant can discover and automatically configure `zeroconf`/`mDNS` and `UPnP` devices and add-ons on your network. In order for this to work you must create the container with `--net=host`:

when using `docker cli`:
```sh
--net=host
```

when using `docker-compose.yaml`:
```yaml
network_mode: host
```
<hr>
<br>
</details>

<details>
<summary><b>Add-ons auto-discovery</b></summary>
<hr>

> **TLDR;** *It's disabled, go with manual way...*

The native auto-discovery of add-ons running on the same host/network is disabled due to the requirement of running [`Home Assistant Supervisor`](https://www.home-assistant.io/integrations/hassio/). This has some deep debian system dependencies which ware too tidious to port in this project.

> Most Home Assistant add-on's are using [`bashio`](https://github.com/hassio-addons/bashio) under the hood so some of the system overlays commands ware adjusted to make it work without `Supervisor`.

#### Manual `wyoming` add-on discovery

To manually add the `wyoming` enabled add-on from this repository to the running Home Assistant Core instance, just follow below steps:

1. Browse to your **Home Assistant** instance (eg.: `homeassistant.local:8123`).
2. Go to `Settings > Devices & Services`.
3. In the bottom right corner, select the `Add Integration` button.
4. From the list, search & select `Wyoming Protocol`.
5. Enter the `wyoming` add-on `Host IP` address (use `localhost` if running of the same host as Home Assistant).
6. Enter the `wyoming` add-on `port` (default is `10400`).
<hr>
<br>
</details>

<details>
<summary><b>Accessing Bluetooth Devices</b></summary>
<hr>

In order to provide **Home Assistant** with access to the host's `Bluetooth` device(s), Home Assistant Core container uses `BlueZ` on the `host` - add the capabilities `NET_ADMIN` and `NET_RAW` to the container, and map `dbus` as a `volume` as shown in the below examples to enable Bluetooth support:

when using `docker cli`:
```sh
--cap-add=NET_ADMIN \
--cap-add=NET_RAW \
-v /var/run/dbus:/var/run/dbus:ro
```
when using `docker-compose.yaml`:
```yaml
cap_add:
  - NET_ADMIN
  - NET_RAW
volumes:
  - /var/run/dbus:/var/run/dbus:ro
```
<hr>
<br>
</details>

### TODO's

- [ ] Fix add-ons auto-discovery

### Support

Got questions? You have several options to get them answered:

#### For general **Home Assistant** Support:
- The [Home Assistant Discord Chat Server](https://discord.gg/c5DvZ4e).
- The Home Assistant [Community Forum](https://community.home-assistant.io/).
- Join the [Reddit subreddit](https://reddit.com/r/homeassistant) in [`/r/homeassistant`](https://reddit.com/r/homeassistant)
- In case you've found an bug in Home Assistant, please [open an issue on our GitHub](https://github.com/home-assistant/addons/issues).

#### For NVIDIA Jetson based Home Assistant Support:
- The NVIDIA Jetson AI Lab [tutorials section](https://www.jetson-ai-lab.com/tutorial-intro.html).
- The Jetson AI Lab - Home Assistant Integration [thread on NVIDIA's Developers Forum](https://forums.developer.nvidia.com/t/jetson-ai-lab-home-assistant-integration/288225).
- In case you've found an bug in `jetson-containers`, please [open an issue on our GitHub](https://github.com/dusty-nv/jetson-containers/issues).
