# System Setup Scripts

This directory contains essential system configuration scripts for optimizing Jetson devices for containerized AI/ML workloads.

## Overview

These scripts handle critical system setup tasks including NVMe storage configuration, Docker optimization, memory management, and performance tuning. They are designed specifically for NVIDIA Jetson platforms and use Jetson-specific tools.

## Scripts Description

### setup-system.sh
**Purpose**: Comprehensive system configuration script for initial Jetson setup.

**Key Functions**:
- Configure and mount NVMe drives for expanded storage
- Setup Docker with NVIDIA runtime support
- Relocate Docker data root from eMMC to NVMe
- Configure swap files for additional virtual memory
- Disable zram to prevent memory compression overhead
- Enable/disable desktop GUI to save RAM
- Set power mode for optimal performance
- Add users to docker group for permissions

**Usage**:
```bash
# Full system setup
sudo ./setup-system.sh

# Test specific function
sudo ./setup-system.sh --test=configure_nvme
sudo ./setup-system.sh --test=configure_docker
```

### probe-system.sh
**Purpose**: Diagnostic tool to check system configuration without making changes.

**Key Functions**:
- Verify NVMe mount status and available space
- Check Docker runtime configuration
- Validate Docker data root location
- Report swap file and zram status
- Check GUI configuration
- Verify Docker group membership
- Display current power mode

**Usage**:
```bash
# Check all system settings
sudo ./probe-system.sh

# Check specific component
sudo ./probe-system.sh --test=check_nvme
sudo ./probe-system.sh --test=check_docker
```

### configure-ssd-docker.sh
**Purpose**: Specialized script for Docker and NVMe SSD setup.

**Key Functions**:
- Detect if L4T is installed on NVMe or eMMC
- Format and mount NVMe drives automatically
- Install Docker if not present
- Migrate existing Docker data from eMMC to NVMe
- Configure Docker daemon with NVIDIA runtime

**Usage**:
```bash
# Automatic Docker + NVMe setup
sudo ./configure-ssd-docker.sh
```

### optimize-ram-usage.sh
**Purpose**: Memory optimization for resource-constrained scenarios.

**Key Functions**:
- Toggle desktop GUI (saves ~1GB RAM when disabled)
- Configure ZRAM (compressed swap in RAM)
- Setup swap files on disk
- Set MAXN power mode for best performance
- Display before/after RAM usage

**Usage**:
```bash
# Interactive mode
sudo ./optimize-ram-usage.sh

# Command-line options
sudo ./optimize-ram-usage.sh --disable-gui --enable-swap
sudo ./optimize-ram-usage.sh --enable-gui --disable-zram
```

## Configuration Files

### .env
Environment variables for script configuration:
```bash
# Example .env file
NVME_MOUNT=/mnt/nvme
DOCKER_ROOT=/mnt/nvme/docker
SWAP_SIZE=32G
```

### system-config.yaml
YAML configuration file (reserved for future use)

## Typical Workflow

### Initial Setup (First Time)
```bash
# 1. Check current system state
sudo ./probe-system.sh

# 2. Run comprehensive setup
sudo ./setup-system.sh

# 3. Verify changes
sudo ./probe-system.sh

# 4. Reboot to apply all changes
sudo reboot
```

### Storage Expansion (Adding NVMe)
```bash
# 1. Install NVMe drive physically
# 2. Run Docker migration
sudo ./configure-ssd-docker.sh

# 3. Verify Docker is using NVMe
docker info | grep "Docker Root Dir"
```

### Memory Optimization
```bash
# When running out of memory
sudo ./optimize-ram-usage.sh --disable-gui --enable-swap

# For development with GUI needs
sudo ./optimize-ram-usage.sh --enable-gui
```

## Important Notes

### Requirements
- **Sudo/root permissions** required for all scripts
- **Jetson-specific tools**: nvpmodel, jetson_clocks, nvzramconfig
- **Standard Linux tools**: systemctl, mount, mkfs.ext4, fallocate

### Safety Features
- Scripts check current state before making changes
- Automatic backup of critical files (e.g., /etc/docker/daemon.json)
- User confirmation prompts for destructive operations
- Validation after each change

### Platform Support
- Tested on JetPack 5.x and 6.x
- Supports all Jetson platforms (Nano, Xavier, Orin)
- Ubuntu 20.04, 22.04, and 24.04

### Common Use Cases

| Scenario | Solution | Script |
|----------|----------|--------|
| Low storage on eMMC | Move Docker to NVMe | `configure-ssd-docker.sh` |
| Out of memory errors | Disable GUI, add swap | `optimize-ram-usage.sh` |
| Docker permission denied | Add user to docker group | `setup-system.sh` |
| Poor performance | Set MAXN power mode | `setup-system.sh` |
| Need system status | Check configuration | `probe-system.sh` |

## Troubleshooting

### Docker not starting after setup
```bash
# Check Docker service
sudo systemctl status docker

# Verify data root exists
ls -la $(docker info | grep "Docker Root Dir" | awk '{print $4}')

# Re-run setup
sudo ./setup-system.sh --test=configure_docker
```

### NVMe not detected
```bash
# Check if NVMe is visible
lsblk | grep nvme

# Check mount status
mount | grep nvme

# Re-run NVMe setup
sudo ./setup-system.sh --test=configure_nvme
```

### Script fails with permission error
```bash
# Ensure running with sudo
sudo ./script-name.sh

# Check script permissions
chmod +x *.sh
```