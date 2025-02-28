# Welcome to the Scripts Folder

Before running any scripts, please ensure you have a properly configured `.env` file in the project root. This file defines environment variables that control the behavior of the scripts. For example, you might need to set:
- `NVME_SETUP_OPTIONS_MOUNT_POINT`: Mount point for the NVMe drive.
- `DOCKER_RUNTIME_SHOULD_RUN`: Flag for Docker runtime configuration.
- `SWAP_OPTIONS_SIZE`: Desired swap file size.
- ...and other necessary variables.

## Available Scripts

### Probe Script (`probe-system.sh`)
- **What it does:**  
  Checks your system settings like NVMe drive mounting, Docker runtime setup, swap file, zram status, GUI configuration, and more.
- **How to use it:**  
  Run `./probe-system.sh` to get a status overview. For specific checks, use the `--tests` flag, e.g.:  
  `./probe-system.sh --tests=nvme_mount,docker_runtime`

### Setup Script (`setup-system.sh`)
- **What it does:**  
  Configures your system based on your environment settings. It handles tasks such as setting up the NVMe drive, configuring Docker, creating the swap file, and adjusting power modes.
- **How to use it:**  
  Run `./setup-system.sh` as root (e.g., with `sudo`) and follow the step-by-step prompts. The script reads its configuration from your `.env` file and verifies each configuration step by using the probe script.

This document should assist users in understanding the roles and interactions between the probe and setup scripts in configuring the Jetson device.
