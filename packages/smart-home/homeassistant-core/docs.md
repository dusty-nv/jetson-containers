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
