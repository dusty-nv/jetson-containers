## GitHub Actions: self-hosted runner setup (Jetson)

### Follow "Add new self-hosted runner" instruction on GitHub repo

Go to the **Acitons** >> **Runeers** in the repo's **Settings** tab.

https://github.com/NVIDIA-AI-IOT/jetson-containers/settings/actions/runners/new

To add a Jetson as a self-hosted runner:

1. Select **Linux** for "Runner image"
2. Select **ARM64** for "Architecture"
3. Follow and execute the commands listed under "**Downlaod**" section (You can copy each command on the page linked above)

It would go something like this:

> ```bash
> jetson@jat03-iso0807:~$ mkdir actions-runner && cd actions-runner
> jetson@jat03-iso0807:~/actions-runner$ curl -o actions-runner-linux-arm64-2.328.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.328.0/actions-runner-linux-arm64-2.328.0.tar.gz
>   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
>                                  Dload  Upload   Total   Spent    Left  Speed
>   0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
> 100  133M  100  133M    0     0  96.1M      0  0:00:01  0:00:01 --:--:--  112M
> jetson@jat03-iso0807:~/actions-runner$ echo "b801b9809c4d9301932bccadf57ca13533073b2aa9fa9b8e625a8db905b5d8eb  actions-runner-linux-arm64-2.328.0.tar.gz" | shasum -a 256 -c
> actions-runner-linux-arm64-2.328.0.tar.gz: OK
> jetson@jat03-iso0807:~/actions-runner$ tar xzf ./actions-runner-linux-arm64-2.328.0.tar.gz
> ```

4. Follow and execute commands listed under "**Configure**" section.

> [!TIP]
> The command includes the token specific for your repo, so be sure to copy the command block from the GitHub repo's Settings page.

When you run `./config.sh` command, it will ask you a series of questions.

> ```bash
> jetson@jat03-iso0807:~/actions-runner$ ./config.sh --url https://github.com/NVIDIA-AI-IOT/jetson-containers --token AGEQxxxxxxxxxxxxxxxxxxxxxxxxx
>
> --------------------------------------------------------------------------------
> |        ____ _ _   _   _       _          _        _   _                      |
> |       / ___(_) |_| | | |_   _| |__      / \   ___| |_(_) ___  _ __  ___      |
> |      | |  _| | __| |_| | | | | '_ \    / _ \ / __| __| |/ _ \| '_ \/ __|     |
> |      | |_| | | |_|  _  | |_| | |_) |  / ___ \ (__| |_| | (_) | | | \__ \     |
> |       \____|_|\__|_| |_|\__,_|_.__/  /_/   \_\___|\__|_|\___/|_| |_|___/     |
> |                                                                              |
> |                       Self-hosted runner registration                        |
> |                                                                              |
> --------------------------------------------------------------------------------
>
> # Authentication
>
>
> √ Connected to GitHub
>
> # Runner Registration
>
> Enter the name of the runner group to add this runner to: [press Enter for Default]
>
> Enter the name of runner: [press Enter for jat03-iso0807]
>
> This runner will have the following labels: 'self-hosted', 'Linux', 'ARM64'
> Enter any additional labels (ex. label-1,label-2): [press Enter to skip] jetson,thor
>
> √ Runner successfully added
> √ Runner connection is good
>
> # Runner settings
>
> Enter name of work folder: [press Enter for _work]
>
> √ Settings Saved.
>
> jetson@jat03-iso0807:~/actions-runner$
> ```


### Add `sudoers` file for whitelisting some commands.

One issue is that we need to run the `install.sh` script of in the root of this `jetson-containers` repo for the dependency installation, and it requires sudo privileges.
We (may) also need to execute other commands like `rm -rf` to avoid running into permission issue.

One solution is to allow only the specific commands that need sudo to run without a password.

Below are the steps to achieve this.
Please note this has to be run as a part of the setup for the runner on the first time.

#### 1) Create sudoers drop-in (least privilege)
```bash
sudo visudo -f /etc/sudoers.d/jetson-actions
```

> If you want to specify VIM to be the editor
> ```bash
> sudo EDITOR=vim visudo -f /etc/sudoers.d/jetson-actions
> ```


Add (adjust paths if they differ from your system defaults):
```text
$USER ALL=(ALL) NOPASSWD: /usr/bin/apt-get, /usr/bin/ln, /bin/rm -rf /home/jetson/actions-runner/_work/*, /bin/chmod -R /home/jetson/actions-runner/_work/*, /bin/chown -R /home/jetson/actions-runner/_work/*
```

> [!IMPORTANT]
> `$USER` is not expanded by `visudo`. To create a username-agnostic entry non-interactively that resolves your current user at creation time, you can instead run:
>
> ```bash
> sudo tee /etc/sudoers.d/jetson-actions >/dev/null <<EOF
> $USER ALL=(ALL) NOPASSWD: /usr/bin/apt-get, /usr/bin/ln, /bin/rm -rf /home/jetson/actions-runner/_work/*, /bin/chmod -R /> home/jetson/actions-runner/_work/*, /bin/chown -R /home/jetson/actions-runner/_work/*
> EOF
> ```

#### 2) Fix ownership and permissions
```bash
sudo chown root:root /etc/sudoers.d/jetson-actions
sudo chmod 0440 /etc/sudoers.d/jetson-actions
```

#### 3) Validate
```bash
sudo visudo -c
sudo -n -l | grep -E 'apt-get|ln'
sudo -n apt-get -h >/dev/null
sudo -n ln --help >/dev/null
```


> [!NOTE]
> - To revert: `sudo rm /etc/sudoers.d/jetson-actions`.


### Add your user to the docker group

You need to add your user to the docker group so `docker` doesn't require sudo.

```bash
# ensure the docker group exists (no-op if it already does)
sudo groupadd -f docker

# add current user to docker group
sudo usermod -aG docker $USER

# activate group without full logout (current shell only)
newgrp docker

# validate: these should run without sudo
docker info | sed -n '1,12p' | cat
docker run --rm hello-world | cat
```

If you still see permission errors, log out and log back in (or reboot) to refresh group membership. This prevents tools like `container.py` from needing to prefix `sudo` for docker commands.



### Set up runner service

You can create a service so that the runner programs automatically starts on boot.

Assuming you have already downloaded and extracted the runner into a directory (e.g., `~/actions-runner`):

```bash
cd ~/actions-runner

# install as a systemd service
sudo ./svc.sh install

# control the service
sudo ./svc.sh start
sudo ./svc.sh status
# sudo ./svc.sh stop
# sudo ./svc.sh restart
# sudo ./svc.sh uninstall
```

To check the service's log in realtime (by following), you can use `journalctl` command.

```bash
journalctl -u actions.runner.NVIDIA-AI-IOT-jetson-containers.jat02-iso382.service -f
```


Logs and troubleshooting with `journalctl`:

- Service unit name is typically:
  - Repo-level: `actions.runner.<org>-<repo>.<runner>.service`
  - Org-level: `actions.runner.<org>.<runner>.service`

```bash
# show recent logs (adjust the unit to your actual values)
sudo journalctl -u actions.runner.<org>-<repo>.<runner>.service -n 200 --no-pager

# follow logs live
sudo journalctl -u actions.runner.<org>-<repo>.<runner>.service -f

# alternatively, try to discover the unit name automatically
UNIT=$(systemctl list-units --type=service --all | awk '/actions\.runner/ {print $1; exit}')
echo "Detected unit: $UNIT"
sudo journalctl -u "$UNIT" -n 200 --no-pager
```

