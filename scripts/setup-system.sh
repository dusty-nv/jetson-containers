#!/bin/bash

# Enable error handling
set -euo pipefail

# Initialize variables
declare -a TESTS=()

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load environment variables from .env
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
elif [ -f "${SCRIPT_DIR}/../.env" ]; then
    set -a
    source "${SCRIPT_DIR}/../.env"
    set +a
else
    echo "Environment file .env not found."
    exit 1
fi

# Initialize configuration flags with defaults
interactive_mode="${INTERACTIVE_MODE:-true}"

# NVMe Setup - Only set variables if NVME_SETUP_SHOULD_RUN is defined in .env
if [ -n "${NVME_SETUP_SHOULD_RUN+x}" ]; then
    nvme_should_run="${NVME_SETUP_SHOULD_RUN:-no}"
    MOUNT_POINT="${NVME_SETUP_OPTIONS_MOUNT_POINT:-/mnt}"
    partition_name="${NVME_SETUP_OPTIONS_PARTITION_NAME:-nvme0n1}"
    filesystem="${NVME_SETUP_OPTIONS_FILESYSTEM:-ext4}"
fi

# Docker Runtime
docker_runtime_should_run="${DOCKER_RUNTIME_SHOULD_RUN:-no}"

# Docker Root
docker_root_should_run="${DOCKER_ROOT_SHOULD_RUN:-no}"
docker_root_path="${DOCKER_ROOT_OPTIONS_PATH:-/mnt/docker}"

# Swap
swap_should_run="${SWAP_SHOULD_RUN:-no}"
disable_zram_flag="${SWAP_OPTIONS_DISABLE_ZRAM:-true}"
swap_size="${SWAP_OPTIONS_SIZE:-32}"
swap_file="${SWAP_OPTIONS_PATH:-/mnt/32GB.swap}"

# GUI
gui_disabled_should_run="${GUI_DISABLED_SHOULD_RUN:-ask}"

# Docker Group
docker_group_should_run="${DOCKER_GROUP_SHOULD_RUN:-no}"
add_user="${DOCKER_GROUP_OPTIONS_ADD_USER:-jetson}"

# Power Mode
power_mode_should_run="${POWER_MODE_SHOULD_RUN:-ask}"
power_mode="${POWER_MODE_OPTIONS_MODE:-1}"

# Check if script is run with sudo
check_permissions() {
    if [ "$EUID" -ne 0 ]; then 
        echo "Please run as root (with sudo)"
        exit 1
    fi
}

# Check for minimal dependencies
check_dependencies() {
    local missing_deps=()
    
    # Required commands
    if ! command -v nvpmodel &> /dev/null; then
        missing_deps+=("nvpmodel")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo "Error: Missing required dependencies: ${missing_deps[*]}"
        echo "Please install them using:"
        echo "sudo apt-get install ${missing_deps[*]}"
        exit 1
    fi
}

# Function to prompt yes/no questions
ask_yes_no() {
    while true; do
        read -p "$1 (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or y or no or n.";;
        esac
    done
}

# Function to assign and mount NVMe drive
assign_nvme_drive() {
    mount_point="$MOUNT_POINT"
    partition_name="$partition_name"
    filesystem="$filesystem"

    if mount | grep -q "/dev/$partition_name on $mount_point"; then
        echo "NVMe is already mounted on $mount_point, skipping assignment."
        return 0
    fi

    echo "Creating mount point if it doesn't exist..."
    if [ ! -d "$mount_point" ]; then
        mkdir -p "$mount_point" || {
            echo "Failed to create mount point"
            return 1
        }
    fi

    if ! grep -q "/dev/$partition_name" /etc/fstab; then
        echo "Adding NVMe mount to fstab..."
        echo "/dev/$partition_name $mount_point $filesystem defaults 0 0" >> /etc/fstab || {
            echo "Failed to update fstab"
            return 1
        }
    fi

    echo "Mounting NVMe drive..."
    if mount "/dev/$partition_name"; then
        echo "NVMe drive mounted successfully."
        return 0
    else
        echo "Failed to mount NVMe drive."
        return 1
    fi
}

# Check if Docker is installed
is_docker_installed() {
    if command -v docker &>/dev/null; then
        return 0 # true, Docker is installed
    else
        return 1 # false, Docker is not installed
    fi
}

# Install Docker and NVIDIA container toolkit
install_docker() {
    echo "Installing Docker and NVIDIA container toolkit..."
    
    # Update package lists
    apt-get update || {
        echo "Failed to update package lists"
        return 1
    }
    
    # Install Docker packages
    if ! apt-get install -y docker.io docker-compose || ! systemctl enable docker; then
        echo "Failed to install Docker"
        return 1
    fi
    
    # Install NVIDIA container packages for Jetson
    if ! apt-get install -y nvidia-container nvidia-container-toolkit; then
        echo "Failed to install NVIDIA container toolkit"
        return 1
    fi
    
    # Restart Docker to apply changes
    if ! systemctl restart docker; then
        echo "Failed to restart Docker service"
        return 1
    fi
    
    echo "Docker and NVIDIA container toolkit installed successfully"
    return 0
}

# Configure Docker runtime
setup_docker_runtime() {
    local daemon_json="/etc/docker/daemon.json"
    local dateFileVersionFormat=$(date +"%Y%m%d%H%M%S")
    local daemon_json_backup="${daemon_json}.${dateFileVersionFormat}.bak"

    # Check if Docker is installed
    if ! is_docker_installed; then
        echo "Docker is not installed. Please install Docker first."
        return 1
    fi

    # Check if daemon.json exists, create if not
    if [ ! -f "$daemon_json" ]; then
        echo "Docker daemon.json file not found, creating with default NVIDIA runtime configuration..."
        mkdir -p "$(dirname $daemon_json)"
        cat > "$daemon_json" << 'EOF'
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
    else
        echo "Using existing daemon.json configuration."
    fi

    # Don't skip if already configured - we need to verify if it's working
    # Create backup
    cp "$daemon_json" "$daemon_json_backup" || {
        echo "Failed to create backup of daemon.json"
        return 1
    }

    echo "Configuring Docker runtime..."
    if ! grep -q '"runtimes"' "$daemon_json"; then
        # Add complete nvidia runtime configuration
        cat > "$daemon_json" << 'EOF'
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF
    else
        # Add nvidia runtime if runtimes section exists but nvidia runtime doesn't
        if ! grep -q '"nvidia"' "$daemon_json"; then
            # Add nvidia runtime to runtimes section
            sed -i '/runtimes/a \        "nvidia": {\n            "path": "nvidia-container-runtime",\n            "runtimeArgs": []\n        },' "$daemon_json"
        fi
        
        # Remove any existing default-runtime line to avoid duplicates
        sed -i '/"default-runtime"/d' "$daemon_json"
        
        # Add default-runtime before the closing brace
        sed -i '0,/}/ s/}$/,\n    "default-runtime": "nvidia"\n}/' "$daemon_json"
    fi

    if ! grep -q '"default-runtime": "nvidia"' "$daemon_json"; then
        echo "Failed to configure Docker runtime."
        mv "$daemon_json_backup" "$daemon_json"
        return 1
    else
        echo "Docker runtime configured successfully."
        # Restart Docker to apply changes
        systemctl restart docker
        return 0
    fi
}

# Configure Docker data root
setup_docker_root() {
    docker_root="$docker_root_path"

    if ! is_docker_installed; then
        echo "Docker is not installed. Please install Docker first."
        return 1
    fi

    if [ -d "/mnt/docker" ]; then
        echo "Docker root already configured, skipping..."
        return 0
    fi

    if should_execute_step "docker_root" "Would you like to relocate Docker data root?"; then
        echo "Using Docker root path: $docker_root"
        
        if [ ! -d "$docker_root" ]; then
            echo "Creating directory: $docker_root"
            mkdir -p "$docker_root" || {
                echo "Failed to create directory $docker_root"
                return 1
            }
        fi

        echo "Copying existing Docker data..."
        if ! cp -r /var/lib/docker/* "$docker_root/"; then
            echo "Failed to copy Docker data"
            return 1
        fi

        # Update daemon.json
        local daemon_json="/etc/docker/daemon.json"
        if ! sed -i 's/}\s*$/,\n    "data-root": "'"$docker_root"'"\n}/' "$daemon_json"; then
            echo "Failed to update daemon.json"
            return 1
        fi

        return 0
    fi
    return 0
}

# Setup swap file
setup_swap_file() {
    swap_size="$swap_size"
    swap_file="$swap_file"

    if swapon --show | grep -q "$swap_file"; then
        echo "Swap already configured, skipping..."
        return 0
    fi

    if should_execute_step "swap_file" "Configure swap file"; then
        echo "Setting up swap with size: ${swap_size}GB at: $swap_file"

        # Create parent directory if it doesn't exist
        local swap_dir=$(dirname "$swap_file")
        if [ ! -d "$swap_dir" ]; then
            echo "Creating swap parent directory: $swap_dir"
            mkdir -p "$swap_dir" || {
                echo "Failed to create swap directory"
                return 1
            }
        fi

        echo "Creating swap file..."
        if ! fallocate -l "${swap_size}G" "$swap_file" || \
           ! chmod 600 "$swap_file" || \
           ! mkswap "$swap_file" || \
           ! swapon "$swap_file"; then
            echo "Failed to setup swap"
            return 1
        fi
            
        if ! grep -q "$swap_file" /etc/fstab; then
            echo "$swap_file  none  swap  sw 0  0" >> /etc/fstab || {
                echo "Failed to update fstab"
                return 1
            }
        fi
            
        echo "Swap file configured successfully."
        return 0
    fi
    return 0
}

# Disable zram
disable_zram() {
    if ! systemctl is-enabled nvzramconfig &> /dev/null; then
        echo "zram is already disabled, skipping..."
        return 0
    fi

    if should_execute_step "disable_zram" "Disable zram (nvzramconfig)"; then
        echo "Disabling zram (nvzramconfig)..."
        if systemctl disable nvzramconfig && systemctl stop nvzramconfig; then
            echo "zram disabled successfully."
            return 0
        else
            echo "Failed to disable zram."
            return 1
        fi
    fi
    return 0
}

# Configure desktop GUI
setup_gui() {
    if should_execute_step "gui_disabled" "Would you like to disable the desktop GUI on boot?"; then
        if systemctl set-default multi-user.target; then
            echo "Desktop GUI disabled successfully."
            return 0
        else
            echo "Failed to configure GUI"
            return 1
        fi
    else
        if systemctl set-default graphical.target; then
            echo "Desktop GUI enabled successfully."
            return 0
        else
            echo "Failed to configure GUI"
            return 1
        fi
    fi
    return 0
}

# Configure Docker group
setup_docker_group() {
    if ! is_docker_installed; then
        echo "Docker is not installed. Please install Docker first."
        return 1
    fi
    
    local configured_user="$add_user"
    local current_user=$(who am i | awk '{print $1}')
    [ -z "$current_user" ] && current_user=$(logname 2>/dev/null || echo "$SUDO_USER")
    local users_to_add=()
    
    # Check the configured user from .env if provided
    if [ -n "$configured_user" ]; then
        if ! getent group docker | grep -q "\b${configured_user}\b"; then
            users_to_add+=("$configured_user")
        else
            echo "User $configured_user is already in the docker group, skipping..."
        fi
    fi
    
    # Check the current user if different from the configured user or if no user configured
    if [ "$current_user" != "root" ] && ([ -z "$configured_user" ] || [ "$current_user" != "$configured_user" ]); then
        if ! getent group docker | grep -q "\b${current_user}\b"; then
            users_to_add+=("$current_user")
        else
            echo "Current user $current_user is already in the docker group, skipping..."
        fi
    fi
    
    # If no users need to be added, exit early
    if [ ${#users_to_add[@]} -eq 0 ]; then
        echo "All relevant users are already in the docker group, skipping..."
        return 0
    fi
    
    echo "Adding user(s) to the docker group..."
    local failed=false
    
    for user in "${users_to_add[@]}"; do
        echo "Adding user $user to docker group..."
        if ! usermod -aG docker "$user"; then
            echo "Failed to add user $user to Docker group"
            failed=true
        else
            echo "Successfully added $user to docker group"
        fi
    done
    
    # Verify the changes
    for user in "${users_to_add[@]}"; do
        if ! getent group docker | grep -q "\b${user}\b"; then
            echo "Verification failed: User $user was not added to the docker group"
            return 1
        fi
    done
    
    echo "Docker group permissions updated successfully. Changes will take effect after user logout/login or system reboot."
    return 0
}

# Configure power mode
setup_power_mode() {
    local mode="$power_mode"
    local current_mode=$(nvpmodel -q | grep "NV Power Mode" | cut -d':' -f2 | xargs)
    
    if [ "$current_mode" = "MODE_$mode" ]; then
        echo "Power mode already set to MODE_$mode, skipping..."
        return 0
    fi

    if should_execute_step "power_mode" "Would you like to set the power mode? (May require reboot)"; then
        echo "Setting power mode to mode $mode (this will be applied after reboot)..."
        
        # Use -f flag to suppress the interactive reboot prompt
        if nvpmodel -m "$mode" -f; then
            echo "Power mode change scheduled. A reboot will be required to apply this change."
            power_mode_changed=true
            return 0
        else
            echo "Failed to set power mode"
            return 1
        fi
    fi
    return 0
}

# Ask for execution mode
ask_execution_mode() {
    echo "Choose execution mode:"
    echo "1) Run all configuration steps"
    echo "2) Select steps individually"
    while true; do
        read -p "Enter choice (1/2): " choice
        case $choice in
            1) 
                execution_mode="all"
                return 0
                ;;
            2)
                execution_mode="individual"
                return 0
                ;;
            *) echo "Please enter 1 or 2.";;
        esac
    done
}

# Function to parse command-line arguments
parse_args() {
    TESTS=()
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --tests)
                IFS=',' read -ra TESTS <<< "$2"
                shift 2
                ;;
            --tests=*)
                IFS=',' read -ra TESTS <<< "${1#*=}"
                shift
                ;;
            *)
                echo "Unknown parameter passed: $1"
                exit 1
                ;;
        esac
    done
}

# Check if specific tests are provided
if [ ${#TESTS[@]} -ne 0 ]; then
    execution_mode="specific"
else
    ask_execution_mode
fi

# Check if step should be executed based on mode
should_execute_step() {
    local step=$1
    local prompt=$2
    
    if [ "$execution_mode" = "specific" ]; then
        for test in "${TESTS[@]}"; do
            if [ "$test" = "$step" ]; then
                return 0
            fi
        done
        return 1
    elif [ "$execution_mode" = "all" ]; then
        return 0
    else
        ask_yes_no "$prompt"
    fi
}

# Load environment variables from .env
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
else
    echo "Environment file .env not found."
    exit 1
fi


# Main execution
main() {
    echo "=== Jetson Setup Script ==="
    echo "This script will help configure your Jetson device."
    echo "You will need to reboot once after completion."
    echo

    check_permissions
    check_dependencies
    echo "Probing status before working"
    echo "============================="
    "${SCRIPT_DIR}/probe-system.sh"
    echo "============================="

    # Check Docker Installation
    docker_installed=false
    if is_docker_installed; then
        docker_installed=true
        echo "Docker is already installed."
    else
        echo "Docker is not installed."
        if [ "$interactive_mode" = "true" ] && ask_yes_no "Install Docker and NVIDIA container runtime?"; then
            install_docker
            if is_docker_installed; then
                docker_installed=true
                echo "Docker installed successfully."
            else
                echo "Failed to install Docker. Please install it manually."
            fi
        fi
    fi

    # NVMe Setup - Only run if NVME_SETUP_SHOULD_RUN is defined in .env
    if [ -n "${NVME_SETUP_SHOULD_RUN+x}" ]; then
        if [ "$nvme_should_run" = "yes" ] || ([ "$nvme_should_run" = "ask" ] && [ "$interactive_mode" = "true" ] && ask_yes_no "Configure NVMe drive?"); then
            if ! "${SCRIPT_DIR}/probe-system.sh" --tests="nvme_mount"; then
                assign_nvme_drive
                "${SCRIPT_DIR}/probe-system.sh" --tests="nvme_mount"
            else
                echo "NVMe drive is already properly mounted."
            fi
        fi
    fi

    # Docker Runtime Setup
    if $docker_installed; then
        if [ "$docker_runtime_should_run" = "yes" ] || ([ "$docker_runtime_should_run" = "ask" ] && [ "$interactive_mode" = "true" ] && ask_yes_no "Configure Docker runtime?"); then
            echo "=== System Probe Script === (Docker Runtime before configuration)"
            "${SCRIPT_DIR}/probe-system.sh" --tests="docker_runtime"
            setup_docker_runtime
            echo "=== System Probe Script === (Docker Runtime after configuration)"
            "${SCRIPT_DIR}/probe-system.sh" --tests="docker_runtime"
        fi
    fi

    # Docker Root Setup
    if $docker_installed; then
        if [ "$docker_root_should_run" = "yes" ] || ([ "$docker_root_should_run" = "ask" ] && [ "$interactive_mode" = "true" ] && ask_yes_no "Configure Docker root?"); then
            echo "=== System Probe Script === (Docker Root before configuration)"
            "${SCRIPT_DIR}/probe-system.sh" --tests="docker_root"
            setup_docker_root
            echo "=== System Probe Script === (Docker Root after configuration)"
            "${SCRIPT_DIR}/probe-system.sh" --tests="docker_root"
        fi
    fi

    # Swap Setup
    if [ "$swap_should_run" = "yes" ] || ([ "$swap_should_run" = "ask" ] && [ "$interactive_mode" = "true" ] && ask_yes_no "Configure swap?"); then
        echo "=== System Probe Script === (Swap File before configuration)"
        "${SCRIPT_DIR}/probe-system.sh" --tests="swap_file"
        setup_swap_file
        echo "=== System Probe Script === (Swap File after configuration)"
        "${SCRIPT_DIR}/probe-system.sh" --tests="swap_file"
    fi

    # ZRAM Setup
    if [ "$disable_zram_flag" = "true" ]; then
        echo "=== System Probe Script === (ZRAM before configuration)"
        "${SCRIPT_DIR}/probe-system.sh" --tests="disable_zram,nvzramconfig_service"
        disable_zram
        echo "=== System Probe Script === (ZRAM after configuration)"
        "${SCRIPT_DIR}/probe-system.sh" --tests="disable_zram,nvzramconfig_service"
    fi

    # GUI Setup
    if [ "$gui_disabled_should_run" = "yes" ] || ([ "$gui_disabled_should_run" = "ask" ] && [ "$interactive_mode" = "true" ] && ask_yes_no "Configure desktop GUI?"); then
        echo "=== System Probe Script === (GUI before configuration)"
        "${SCRIPT_DIR}/probe-system.sh" --tests="gui"
        setup_gui
        echo "=== System Probe Script === (GUI after configuration)"
        "${SCRIPT_DIR}/probe-system.sh" --tests="gui"
    fi

    # Docker Group Setup
    if $docker_installed; then
        if [ "$docker_group_should_run" = "yes" ] || ([ "$docker_group_should_run" = "ask" ] && [ "$interactive_mode" = "true" ] && ask_yes_no "Configure Docker group?"); then
            echo "=== System Probe Script === (Docker Group before configuration)"
            "${SCRIPT_DIR}/probe-system.sh" --tests="docker_group"
            setup_docker_group
            echo "=== System Probe Script === (Docker Group after configuration)"
            "${SCRIPT_DIR}/probe-system.sh" --tests="docker_group"
        fi
    fi

    # Power Mode Setup
    if [ "$power_mode_should_run" = "yes" ] || ([ "$power_mode_should_run" = "ask" ] && [ "$interactive_mode" = "true" ] && ask_yes_no "Configure power mode?"); then
        echo "=== System Probe Script === (Power Mode before configuration)"
        "${SCRIPT_DIR}/probe-system.sh" --tests="power_mode"
        setup_power_mode
        echo "=== System Probe Script === (Power Mode after configuration)"
        "${SCRIPT_DIR}/probe-system.sh" --tests="power_mode"
    fi
    
    # Restart Docker service if needed
    if $docker_installed && should_execute_step "docker_service" "Restart Docker service"; then
        echo "Restarting Docker service..."
        systemctl restart docker
        echo "Docker service restarted."
    fi
    
    echo
    echo "Configuration complete!"
    echo "============================="
    "${SCRIPT_DIR}/probe-system.sh"
    echo "============================="

    echo "Please reboot your system for all changes to take effect."
    if [ "$interactive_mode" = "true" ] && ask_yes_no "Would you like to reboot now?"; then
        reboot
    fi
}

# Run main function with all passed arguments
main "$@"