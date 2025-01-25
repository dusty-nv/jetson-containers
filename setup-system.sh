#!/bin/bash

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
    if ! command -v parted &> /dev/null; then
        missing_deps+=("parted")
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

# Function to prepare NVMe partition
prepare_nvme_partition() {
    mount_point="$MOUNT_POINT"
    partition_name="$partition_name"
    filesystem="$filesystem"

    if [ ! -b "/dev/$partition_name" ]; then
        echo "No partition found (/dev/$partition_name)."
        if ask_yes_no "Would you like to create a new partition on the NVMe drive? (WARNING: This will erase all data)"; then
            echo "Creating partition on NVMe drive..."
            parted /dev/$partition_name mklabel gpt
            parted /dev/$partition_name mkpart primary ext4 0% 100%
            sleep 2  # Wait for partition to be recognized
        else
            echo "Skipping partition creation"
            return 1
        fi
    fi

    if ! blkid "/dev/$partition_name" | grep -q "$filesystem"; then
        echo "Partition needs formatting."
        if ask_yes_no "Would you like to format as $filesystem? (WARNING: This will erase all data)"; then
            echo "Formatting NVMe partition as $filesystem..."
            mkfs.$filesystem "/dev/$partition_name"
        else
            echo "Skipping formatting"
            return 1
        fi
    fi

    return 0
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
        mkdir -p "$mount_point"
    fi

    if ! grep -q "/dev/$partition_name" /etc/fstab; then
        echo "Adding NVMe mount to fstab..."
        echo "/dev/$partition_name $mount_point $filesystem defaults 0 0" >> /etc/docker/daemon.json
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
    if grep -q '"default-runtime": "nvidia"' /etc/docker/daemon.json; then
        echo "Docker runtime already configured, skipping..."
        return 0
    fi

    if should_execute_step "docker_runtime" "Would you like to configure Docker runtime?"; then
        echo "Configuring Docker runtime..."
        
        if grep -q '"default-runtime"' /etc/docker/daemon.json; then
            # Replace existing default-runtime with "nvidia"
            sed -i 's/"default-runtime": *"[^"]*"/"default-runtime": "nvidia"/' /etc/docker/daemon.json
        else
            # Insert "default-runtime": "nvidia" before the closing }
            sed -i '/}/i \    "default-runtime": "nvidia"' /etc/docker/daemon.json
        fi

        # Ensure the JSON is valid by adding a comma if necessary
        # Check if the line before the inserted/default-runtime line ends with a comma
        if ! grep -q '"default-runtime": "nvidia"' /etc/docker/daemon.json; then
            echo "Failed to configure Docker runtime."
            return 1
        fi

        echo "Docker runtime configured successfully."
        return 0
    fi
    return 0
}

# Configure Docker data root
setup_docker_root() {
    # Replace hard-coded variable with loaded configuration
    docker_root="$docker_root_path"

    if [ -d "/mnt/docker" ]; then
        echo "Docker root already configured, skipping..."
        return 0
    fi

    if should_execute_step "docker_root" "Would you like to relocate Docker data root?"; then
        echo "Using Docker root path: $docker_root"
        
        if [ ! -d "$docker_root" ]; then
            echo "Creating directory: $docker_root"
            mkdir -p "$docker_root"
        fi

        if [ -d "$docker_root" ]; then
            echo "Copying existing Docker data..."
            if cp -r /var/lib/docker/* "$docker_root/"; then
                # Update daemon.json
                sed -i 's/}\s*$/,\n    "data-root": "'"$docker_root"'"\n}/' /etc/docker/daemon.json
                return 0
            else
                echo "Failed to copy Docker data"
                return 1
            fi
        else
            echo "Failed to create directory $docker_root"
            return 1
        fi
    fi
    return 0
}

# Modify setup_swap_file to handle only swap file setup
setup_swap_file() {
    # Replace hard-coded variables with loaded configuration
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
            mkdir -p "$swap_dir"
        fi

        echo "Creating swap file..."
        if fallocate -l "${swap_size}G" "$swap_file" && \
           chmod 600 "$swap_file" && \
           mkswap "$swap_file" && \
           swapon "$swap_file"; then
            
            if ! grep -q "$swap_file" /etc/fstab; then
                echo "$swap_file  none  swap  sw 0  0" >> /etc/fstab
            fi
            
            echo "Swap file configured successfully."
            return 0
        else
            echo "Failed to setup swap"
            return 1
        fi
    fi
    return 0
}

# Add a new function to disable zram
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

# Add setup_nvzramconfig_service function
setup_nvzramconfig_service() {
    if check_nvzramconfig_service; then
        echo "nvzramconfig service already installed, skipping..."
        return 0
    fi

    if should_execute_step "nvzramconfig_service" "Install and configure nvzramconfig service"; then
        echo "Installing nvzramconfig service..."
        apt-get update
        apt-get install -y nvzramconfig

        if systemctl enable nvzramconfig && systemctl start nvzramconfig; then
            echo "nvzramconfig service installed and started."
            return 0
        else
            echo "Failed to install or start nvzramconfig service."
            return 1
        fi
    fi
    return 0
}

# Configure desktop GUI
setup_gui() {
    if systemctl get-default | grep -q "multi-user.target"; then
        echo "GUI already configured, skipping..."
        return 0
    fi

    if should_execute_step "gui_disabled" "Would you like to disable the desktop GUI on boot?"; then
        if systemctl set-default multi-user.target; then
            return 0
        else
            echo "Failed to configure GUI"
            return 1
        fi
    else
        systemctl set-default graphical.target
    fi
    return 0
}

# Configure Docker group
setup_docker_group() {
    # Replace hard-coded user with loaded configuration
    if usermod -aG docker "$add_user"; then
        return 0
    else
        echo "Failed to add user to Docker group"
        return 1
    fi

    if groups $(logname) | grep -q "\bdocker\b"; then
        echo "Docker group already configured, skipping..."
        return 0
    fi

    if should_execute_step "docker_group" "Would you like to add $(logname) to the docker group?"; then
        if usermod -aG docker "$(logname)"; then
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
    # Replace hard-coded mode with loaded configuration
    mode="$power_mode"

    if nvpmodel -q | grep -q "NV Power Mode: MAXN"; then
        echo "Power mode already configured, skipping..."
        return 0
    fi

    # Command not found (should run)
    if should_execute_step "power_mode" "Would you like to set the power mode to 25W (recommended performance mode)?"; then
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
else
    ask_execution_mode
fi

# Check if step should be executed based on mode
should_execute_step() {
    local step=$1
    local prompt=$2
    local mode=$execution_mode
    
    if [ "$mode" = "specific" ]; then
        for test in "${TESTS[@]}"; do
            if [ "$test" == "$step" ]; then
                return 0
            fi
        done
        return 1
    elif [ "$mode" = "all" ]; then
        return 0
    else
        ask_yes_no "Would you like to run the $step step?"
    fi
}

# Add call to probe-system.sh for system checks
probe_system() {
    ./probe-system.sh "$@"
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

    parse_args "$@"
    check_permissions
    check_dependencies
    echo "Probing status before working"
    probe_system --tests="nvme_mount,docker_runtime,docker_root,swap_file,disable_zram,nvzramconfig_service,gui,docker_group,power_mode"

    echo "Loading variables"
    # Assign variables from .env
    interactive_mode="$INTERACTIVE_MODE"
    nvme_should_run="$NVME_SETUP_SHOULD_RUN"
    docker_runtime_should_run="$DOCKER_RUNTIME_SHOULD_RUN"
    docker_root_should_run="$DOCKER_ROOT_SHOULD_RUN"
    swap_should_run="$SWAP_SHOULD_RUN"
    gui_disabled_should_run="$GUI_DISABLED_SHOULD_RUN"
    docker_group_should_run="$DOCKER_GROUP_SHOULD_RUN"
    power_mode_should_run="$POWER_MODE_SHOULD_RUN"

    MOUNT_POINT="$NVME_SETUP_OPTIONS_MOUNT_POINT"
    SWAP_FILE="$SWAP_OPTIONS_PATH"

    partition_name="$NVME_SETUP_OPTIONS_PARTITION_NAME"
    filesystem="$NVME_SETUP_OPTIONS_FILESYSTEM"
    docker_root_path="$DOCKER_ROOT_OPTIONS_PATH"
    disable_zram_flag="$SWAP_OPTIONS_DISABLE_ZRAM"
    swap_size="$SWAP_OPTIONS_SIZE"
    add_user="$DOCKER_GROUP_OPTIONS_ADD_USER"
    power_mode="$POWER_MODE_OPTIONS_MODE"

    # Apply configurations based on settings
    if [ "$nvme_should_run" = "yes" ]; then
        # Prepare NVMe partition
        if should_execute_step "prepare_nvme_partition" "Prepare NVMe partition"; then
            prepare_nvme_partition
        fi

        # Assign NVMe drive
        if should_execute_step "assign_nvme_drive" "Assign and mount NVMe drive"; then
            assign_nvme_drive
        fi
    fi

    # Docker Runtime Setup
    if should_execute_step "docker_runtime" "Configure Docker runtime"; then
        setup_docker_runtime
    fi

    # Docker Root Setup
    if should_execute_step "docker_root" "Configure Docker root directory"; then
        setup_docker_root
    fi

    # Swap Setup File
    if [ "$swap_should_run" = "yes" ]; then
        if should_execute_step "swap_file" "Configure swap file"; then
            setup_swap_file
        fi
    fi

    # Disable zram after setting up swap
    if [ "$disable_zram_flag" = "true" ]; then
        if should_execute_step "disable_zram" "Disable zram (nvzramconfig)"; then
            disable_zram
        fi
    fi

    # GUI Setup
    if should_execute_step "gui_disabled" "Configure desktop GUI"; then
        setup_gui
    fi

    # Docker Group Setup
    if should_execute_step "docker_group" "Configure Docker group"; then
        setup_docker_group
    fi

    # Power Mode Setup
    if should_execute_step "power_mode" "Configure power mode"; then
        setup_power_mode
    fi
    
    # Restart Docker service
    if should_execute_step "docker_service" "Restart Docker service"; then
        systemctl restart docker
    fi

    echo
    echo "Configuration complete!"
    echo "Please reboot your system for all changes to take effect."
    if ask_yes_no "Would you like to reboot now?"; then
        reboot
    fi
}

# Run main function with all passed arguments
main "$@"