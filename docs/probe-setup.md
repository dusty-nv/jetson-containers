# Probe and Setup Scripts Documentation

This document explains the two main scripts in the project:

## Probe Script (`probe-system.sh`)
- **Purpose:** Validates system configuration.
- **Checks Performed:**
  - Verifies NVMe drive mount status.
  - Confirms Docker runtime configuration in `/etc/docker/daemon.json`.
  - Checks Docker data root and swap file configuration.
  - Assesses status of zram (nvzramconfig) and desktop GUI settings.
  - Validates user membership in the Docker group.
  - Reviews current power mode.
- **Usage:** Run the probe script to get current configuration status. It supports both a complete run and selective tests by using the `--tests` flag.

## Setup Script (`setup-system.sh`)
- **Purpose:** Configures the system based on specific setup requirements.
- **Features:**
  - Loads configuration from `.env` and a YAML configuration file.
  - Sets up NVMe drive mounting, Docker runtime and data root.
  - Configures swap file and optionally disables zram.
  - Manages desktop GUI settings and user group memberships.
  - Adjusts power mode of the device.
- **Workflow:**
  - Prompts for confirmation for each step (if in interactive mode) or runs them based on environment settings.
  - Calls the probe script after each configuration step to verify changes.
  - Finishes by suggesting a system reboot to apply all changes.
  
## How to Use
1. **Probing the System:**
   - Execute `./probe-system.sh` to see current system configuration.
   - Use `./probe-system.sh --tests=<test1,test2>` to run selected tests.
2. **Running the Setup:**
   - Execute `./setup-system.sh` as root (using sudo) to begin the configuration process.
   - Follow onscreen prompts if running in interactive mode.
   - The script will validate changes by calling the probe script after each configuration step.

This document should assist users in understanding the roles and interactions between the probe and setup scripts in configuring the Jetson device.
