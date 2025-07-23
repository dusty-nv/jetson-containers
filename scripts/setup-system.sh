#!/bin/bash

# Enable error handling
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(dirname "$(dirname "$(readlink -fm "$0")")")"

# Initialize variables
declare -a TESTS=()

# Load environment variables from .env
ENV_FILE="${JETSON_SETUP_ENV_FILE:-$SCRIPT_DIR/.env}"
if [ -f "${ENV_FILE}" ]; then
    set -a
    source "${ENV_FILE}"
    set +a
else
    echo "Environment file ${ENV_FILE} not found."
    exit 1
fi

# Initialize configuration flags with defaults
interactive_mode="${INTERACTIVE_MODE:-true}"

# NVMe Setup
nvme_should_run="${NVME_SETUP_SHOULD_RUN:-no}"
MOUNT_POINT="${NVME_SETUP_OPTIONS_MOUNT_POINT:-/mnt}"
partition_name="${NVME_SETUP_OPTIONS_PARTITION_NAME:-nvme0n1}"
filesystem="${NVME_SETUP_OPTIONS_FILESYSTEM:-ext4}"

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

# Configure Docker runtime
setup_docker_runtime() {
    local daemon_json="/etc/docker/daemon.json"
    local dateFileVersionFormat=$(date +"%Y%m%d%H%M%S")
    local daemon_json_backup="${daemon_json}.${dateFileVersionFormat}.bak"

    if grep -q '"default-runtime": "nvidia"' "$daemon_json"; then
        echo "Docker runtime already configured, skipping..."
        return 0
    fi

    if should_execute_step "docker_runtime" "Would you like to configure Docker runtime?"; then
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
            # Add only default-runtime if runtimes already exists
            sed -i '/}/i \    "default-runtime": "nvidia",' "$daemon_json"
        fi

        if ! grep -q '"default-runtime": "nvidia"' "$daemon_json"; then
            echo "Failed to configure Docker runtime."
            mv "$daemon_json_backup" "$daemon_json"
            return 1
        fi

        echo "Docker runtime configured successfully."
        return 0
    fi
    return 0
}

# Configure Docker data root
setup_docker_root() {
    docker_root="$docker_root_path"

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
    local current_user="$add_user"
    
    if groups "$current_user" | grep -q "\bdocker\b"; then
        echo "User $current_user is already in the docker group, skipping..."
        return 0
    fi

    if should_execute_step "docker_group" "Would you like to add $current_user to the docker group?"; then
        if usermod -aG docker "$current_user"; then
            echo "Successfully added $current_user to docker group"
            return 0
        else
            echo "Failed to add user to Docker group"
            return 1
        fi
    fi
    return 0
}

# Configure power mode
setup_power_mode() {
    mode="$power_mode"

    current_mode=$(nvpmodel -q | grep "NV Power Mode" | cut -d':' -f2 | xargs)
    if [ "$current_mode" = "MAXN" ]; then
        echo "Power mode already set to MAXN, skipping..."
        return 0
    fi

    if should_execute_step "power_mode" "Would you like to set the power mode to MAXN (recommended performance mode)?"; then
        if nvpmodel -m "$mode"; then
            local mode_name=$(nvpmodel -q | grep "NV Power Mode" | cut -d':' -f2 | xargs)
            echo "Power mode set to: $mode_name"
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
elif [ "${interactive_mode}" = "false" ]; then
    # non‑interactive mode: run all
    execution_mode="all"
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
    $SCRIPT_DIR/probe-system.sh
    echo "============================="

    # NVMe Setup
    if [ "$nvme_should_run" = "yes" ] || ([ "$nvme_should_run" = "ask" -a "$interactive_mode" = "true" ] && ask_yes_no "Configure NVMe drive?"); then
        if ! $SCRIPT_DIR/probe-system.sh --quiet --tests="nvme_mount"; then
            assign_nvme_drive
            $SCRIPT_DIR/probe-system.sh --quiet --tests="nvme_mount"
        else
            echo "✅ NVMe drive is already properly mounted."
        fi
    fi

    # Docker Runtime Setup
    if [ "$docker_runtime_should_run" = "yes" ] || ([ "$docker_runtime_should_run" = "ask" -a "$interactive_mode" = "true" ] && ask_yes_no "Configure Docker runtime?"); then
        if ! $SCRIPT_DIR/probe-system.sh --quiet --tests="docker_runtime"; then
            setup_docker_runtime
            $SCRIPT_DIR/probe-system.sh --quiet --tests="docker_runtime"
        else
            echo "✅ Docker runtime is already properly configured."
        fi
    fi

    # Docker Root Setup
    if [ "$docker_root_should_run" = "yes" ] || ([ "$docker_root_should_run" = "ask" -a "$interactive_mode" = "true" ] && ask_yes_no "Configure Docker root?"); then
        if ! $SCRIPT_DIR/probe-system.sh --quiet --tests="docker_root"; then
            setup_docker_root
            $SCRIPT_DIR/probe-system.sh --quiet --tests="docker_root"
        else
            echo "✅ Docker root is already properly configured."
        fi
    fi

    # Swap Setup
    if [ "$swap_should_run" = "yes" ] || ([ "$swap_should_run" = "ask" -a "$interactive_mode" = "true" ] && ask_yes_no "Configure swap?"); then
        if ! $SCRIPT_DIR/probe-system.sh --quiet --tests="swap_file"; then
            setup_swap_file
            $SCRIPT_DIR/probe-system.sh --quiet --tests="swap_file"
        else
            echo "✅ Swap file is already properly configured."
        fi
    fi

    # ZRAM Setup
    if [ "$disable_zram_flag" = "true" ]; then
        if ! $SCRIPT_DIR/probe-system.sh --quiet --tests="disable_zram,nvzramconfig_service"; then
            disable_zram
            $SCRIPT_DIR/probe-system.sh --quiet --tests="disable_zram,nvzramconfig_service"
        else
            echo "✅ ZRAM is already properly disabled."
        fi
    fi

    # GUI Setup
    if [ "$gui_disabled_should_run" = "yes" ] || ([ "$gui_disabled_should_run" = "ask" -a "$interactive_mode" = "true" ] && ask_yes_no "Configure desktop GUI?"); then
        if ! $SCRIPT_DIR/probe-system.sh --quiet --tests="gui"; then
            setup_gui
            $SCRIPT_DIR/probe-system.sh --quiet --tests="gui"
        else
            echo "✅ GUI is already properly configured."
        fi
    fi

    # Docker Group Setup
    if [ "$docker_group_should_run" = "yes" ] || ([ "$docker_group_should_run" = "ask" -a "$interactive_mode" = "true" ] && ask_yes_no "Configure Docker group?"); then
        if ! $SCRIPT_DIR/probe-system.sh --quiet --tests="docker_group"; then
            setup_docker_group
            $SCRIPT_DIR/probe-system.sh --quiet --tests="docker_group"
        else
            echo "✅ Docker group is already properly configured."
        fi
    fi

    # Power Mode Setup
    if [ "$power_mode_should_run" = "yes" ] || ([ "$power_mode_should_run" = "ask" -a "$interactive_mode" = "true" ] && ask_yes_no "Configure power mode?"); then
        if ! $SCRIPT_DIR/probe-system.sh --quiet --tests="power_mode"; then
            setup_power_mode
            $SCRIPT_DIR/probe-system.sh --quiet --tests="power_mode"
        else
            echo "✅ Power mode is already properly configured."
        fi
    fi

    # Restart Docker service
    if should_execute_step "docker_service" "Restart Docker service"; then
        systemctl restart docker
        echo "✅ Restarted Docker service."
    fi
    
    echo
    echo "Configuration complete!"
    echo "============================="
    $SCRIPT_DIR/probe-system.sh
    echo "============================="

    echo "Please reboot your system for all changes to take effect."
    if [ "$interactive_mode" = "true" ] && ask_yes_no "Would you like to reboot now?"; then
        reboot
    fi
}

# Run main function with all passed arguments
main "$@"