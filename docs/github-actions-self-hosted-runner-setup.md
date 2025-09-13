## GitHub Actions: self-hosted runner setup (Jetson)

One issue is that we need to run the `install.sh` script in the root of this repo, and it requires sudo privileges. The solution is to allow only the specific commands that need sudo to run without a password. Below are the steps to achieve this.
Please note this has to be run as a part of the setup for the runner on the first time.

### 1) Create sudoers drop-in (least privilege)
```bash
sudo visudo -f /etc/sudoers.d/jetson-actions
```
Add (adjust paths if they differ from your system defaults):
```text
jetson ALL=(ALL) NOPASSWD: /usr/bin/apt-get, /usr/bin/ln
```

### 2) Fix ownership and permissions
```bash
sudo chown root:root /etc/sudoers.d/jetson-actions
sudo chmod 0440 /etc/sudoers.d/jetson-actions
```

### 3) Validate
```bash
sudo visudo -c
sudo -n -l | grep -E 'apt-get|ln'
sudo -n apt-get -h >/dev/null
sudo -n ln --help >/dev/null
```


### Notes
- To revert: `sudo rm /etc/sudoers.d/jetson-actions`.

