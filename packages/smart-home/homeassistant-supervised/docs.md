## Docker Configuration

Configure Docker: Docker must be set up with specific settings:

**OverlayFS2 Storage**: Ensure Docker uses OverlayFS2 as the storage driver. Verify your storage driver:
```shell
docker info
cat /etc/docker/daemon.json
```

If it's not set to OverlayFS2, you might need to update the Docker configuration:
```shell
sudo nano /etc/docker/daemon.json
```
Add or modify the file as follows:
```json
{
    "storage-driver": "overlay2"
}
```
Restart Docker for changes to take effect:
```shell
sudo systemctl restart docker
```

**Journald Logging Driver**: Configure Docker to use `journald` as the logging driver. Add the following to the `daemon.json` file:
```json
{
    "log-driver": "journald",
    "log-opts": {
        "tag": "{{.Name}}"
    }
}
```
Restart Docker:
```shell
sudo systemctl restart docker
```

**Cgroup v1**: Configure Docker to use cgroup v1. Add the following to the daemon.json file:
```json
{
    "cgroup-version": 1
}
```
Restart Docker:
```shell
sudo systemctl restart docker
```
**Verify Docker Configuration**: After making the changes, verify that Docker is running with the correct configuration:
```shell
docker info
```

## NetworkManager Configuration

Install and Enable NetworkManager:

Install NetworkManager:
```shell
sudo apt install -y network-manager
```
Enable NetworkManager:
```shell
sudo systemctl enable NetworkManager
sudo systemctl start NetworkManager
```
Check NetworkManager Status:
Verify that NetworkManager is running:
```shell
sudo systemctl status NetworkManager
```
## Systemd Journal Gateway

Enable Systemd Journal Gateway:
Ensure the systemd-journal-gatewayd service is installed and enabled:

```shell
sudo systemctl enable systemd-journal-gatewayd
sudo systemctl start systemd-journal-gatewayd
```

Map Journal Gateway to Supervisor:
Create a symbolic link to map the journal gateway socket to the supervisor:
```shell
sudo ln -s /run/systemd-journal-gatewayd.sock /usr/share/hassio/run/systemd-journal-gatewayd.sock
```
