# Probe and Setup Scripts Documentation

This document explains the system configuration scripts in the Jetson Containers project.

## Profile Generation Script (`create-env-profile.sh`)

- **Purpose:** Creates a tailored `.env` configuration file based on your specific Jetson device and storage setup.
- **Features:**
  - Automatically detects your Jetson device type (AGX Orin, Orin Nano)
  - Identifies your storage configuration (eMMC, SD card, NVMe)
  - Generates appropriate configuration settings for Docker, swap, and other system components

### Supported Devices
- **Jetson AGX Orin** - Fully supported with all storage configurations
- **Jetson Orin Nano** - Fully supported with all storage configurations
- **Jetson Orin NX** - Currently not supported by the automatic script

### Usage
```bash
# Generate a configuration profile with auto-detection
./scripts/create-env-profile.sh

# Specify a specific device type
./scripts/create-env-profile.sh --device agx
```

### Available Profiles
- **AGX Orin Profiles**:
  - AGX with OS on eMMC
  - AGX with OS on eMMC + NVMe storage
  - AGX with OS on NVMe
  - AGX with OS on NVMe + additional NVMe

- **Orin Nano Profiles**:
  - Nano with OS on SD card
  - Nano with OS on SD card + NVMe storage
  - Nano with OS on NVMe
  - Nano with OS on NVMe + additional NVMe

## Probe Script (`probe-system.sh`)
- **Purpose:** Validates current system configuration.
- **Checks Performed:**
  - Verifies Docker installation
  - Verifies NVMe drive mount status and partition setup
  - Confirms Docker runtime configuration in `/etc/docker/daemon.json`
  - Checks Docker data root and swap file configuration
  - Assesses status of zram (nvzramconfig) and desktop GUI settings
  - Validates user membership in the Docker group
  - Reviews current power mode

### Usage
```bash
# Run all checks
./scripts/probe-system.sh

# Run specific tests
./scripts/probe-system.sh --tests=<test1,test2,...>
```

### Available Test Options
- `docker_installed` - Check if Docker is installed
- `nvme_mount` - Check if NVMe is mounted correctly
- `prepare_nvme_partition` - Verify if NVMe partition is prepared
- `assign_nvme_drive` - Verify if NVMe drive is assigned/mounted
- `docker_runtime` - Check Docker runtime configuration
- `docker_root` - Check Docker data root configuration
- `swap_file` - Verify swap file configuration
- `disable_zram` - Check if zram is disabled
- `nvzramconfig_service` - Check nvzramconfig service status
- `gui` - Check desktop GUI boot configuration
- `docker_group` - Verify Docker group membership
- `power_mode` - Check current power mode

## Setup Script (`setup-system.sh`)
- **Purpose:** Configures the system based on specific requirements.
- **Configuration Sources:**
  - Loads settings from `.env` file
  - Uses environment variables to determine which actions to take

### Features
- NVMe drive setup and mounting
- Docker runtime configuration (setting nvidia as default)
- Docker data root relocation (typically to NVMe storage)
- Swap file configuration and zram management
- Desktop GUI enable/disable settings
- Docker group membership management
- Power mode optimization

### Execution Modes
1. **All Steps** - Run all configuration steps in sequence
2. **Individual Steps** - Select specific steps to execute
3. **Selective Tests** - Run only specified tests using `--tests=` parameter

### Workflow
- Checks environment configuration and system status
- For each component, verifies current state using probe-system.sh
- Only applies changes where needed
- Validates changes by running probe checks after each modification
- Suggests system reboot upon completion

## Complete System Setup Workflow

The recommended workflow for setting up your Jetson system is:

1. **Generate configuration profile:**
   ```bash
   ./scripts/create-env-profile.sh
   ```

2. **Review and edit the generated .env file if needed**

3. **Apply the configuration:**
   ```bash
   sudo ./scripts/setup-system.sh
   ```

4. **Reboot to apply all changes:**
   ```bash
   sudo reboot
   ```

## Common Configuration Variables (.env)
```
# NVMe Configuration
NVME_SETUP_SHOULD_RUN=yes|no|ask
NVME_SETUP_OPTIONS_MOUNT_POINT=/mnt
NVME_SETUP_OPTIONS_PARTITION_NAME=nvme0n1p1
NVME_SETUP_OPTIONS_FILESYSTEM=ext4

# Docker Configuration
DOCKER_RUNTIME_SHOULD_RUN=yes|no|ask
DOCKER_ROOT_SHOULD_RUN=yes|no|ask
DOCKER_ROOT_OPTIONS_PATH=/mnt/docker

# Swap Configuration  
SWAP_SHOULD_RUN=yes|no|ask
SWAP_OPTIONS_DISABLE_ZRAM=true|false
SWAP_OPTIONS_SIZE=32
SWAP_OPTIONS_PATH=/mnt/32GB.swap

# Other Settings
GUI_DISABLED_SHOULD_RUN=yes|no|ask
DOCKER_GROUP_SHOULD_RUN=yes|no|ask
DOCKER_GROUP_OPTIONS_ADD_USER=$(whoami)
POWER_MODE_SHOULD_RUN=yes|no|ask
POWER_MODE_OPTIONS_MODE=0
INTERACTIVE_MODE=true|false
```

This documentation provides guidance on using the profile generation, probe and setup scripts to configure Jetson devices for optimal container performance.
