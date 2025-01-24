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

# Modify should_run function to check system status
should_run() {
    local action=$1
    local prompt=$2

    case $action in
        "nvme_setup")
            # Example: Check if NVMe is mounted
            if mount | grep -q "/dev/nvme0n1"; then
                return 1
            fi
            ;;
        "docker_runtime")
            if grep -q '"default-runtime": "nvidia"' /etc/docker/daemon.json; then
                return 1
            fi
            ;;
        # Add similar checks for other actions
        *)
            return 0
            ;;
    esac

    # If not found, decide based on prompt
    if [ "$interactive_mode" = "true" ]; then
        ask_yes_no "$2"
        return $?
    else
        return 1
    fi
}

# Setup NVMe if available
setup_nvme() {
    if mount | grep -q "/dev/nvme0n1"; then
        echo "NVMe already configured, skipping..."
        return 0
    fi

    if should_run "nvme_setup" "Would you like to setup the NVMe drive?"; then
        local mount_point="/mnt"
        local partition_name="nvme0n1p1"
        local filesystem="ext4"

        echo "Setting up NVMe drive..."
        
        # Check partition and get user confirmation for any changes
        if [ ! -b "/dev/$partition_name" ]; then
            if ask_yes_no "No partition found (/dev/$partition_name). Would you like to create a new partition on the NVMe drive? (WARNING: This will erase all data)"; then
                echo "Creating partition on NVMe drive..."
                parted /dev/nvme0n1 mklabel gpt
                parted /dev/nvme0n1 mkpart primary ext4 0% 100%
                sleep 2  # Wait for partition to be recognized
            else
                echo "Skipping partition creation"
                return 0
            fi
        fi

        # Also confirm formatting if needed
        if [ -b "/dev/$partition_name" ] && ! blkid "/dev/$partition_name" | grep -q "$filesystem"; then
            if ask_yes_no "Partition needs formatting. Would you like to format as $filesystem? (WARNING: This will erase all data)"; then
                echo "Formatting NVMe partition as $filesystem..."
                mkfs.$filesystem "/dev/$partition_name"
            else
                echo "Skipping formatting"
                return 0
            fi
        fi

        # Create mount point if it doesn't exist
        if [ ! -d "$mount_point" ]; then
            mkdir -p "$mount_point"
        fi

        # Add to fstab if not already present
        if ! grep -q "/dev/$partition_name" /etc/fstab; then
            echo "Adding NVMe mount to fstab..."
            echo "/dev/$partition_name $mount_point $filesystem defaults 0 0" >> /etc/fstab
        fi

        # Mount the drive
        if ! mount | grep -q "/dev/$partition_name"; then
            mount "/dev/$partition_name"
        fi

        return 0
    fi
    return 0
}

# Configure Docker runtime
setup_docker_runtime() {
    if grep -q '"default-runtime": "nvidia"' /etc/docker/daemon.json; then
        echo "Docker runtime already configured, skipping..."
        return 0
    fi

    if should_run "docker_runtime" "Would you like to configure Docker runtime?"; then
        echo "Configuring Docker runtime..."
        cat > /etc/docker/daemon.json <<EOF
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
        return 0
    fi
    return 0
}

# Configure Docker data root
setup_docker_root() {
    if [ -d "/mnt/docker" ]; then
        echo "Docker root already configured, skipping..."
        return 0
    fi

    if should_run "docker_root" "Would you like to relocate Docker data root?"; then
        local docker_root="/mnt/docker"
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

setup_swap() {
    if swapon --show | grep -q "/mnt/16GB.swap"; then
        echo "Swap already configured, skipping..."
        return 0
    fi

    if should_run "swap" "Would you like to configure swap space?"; then
        local disable_zram=true
        local swap_size="16"
        local swap_file="/mnt/16GB.swap"

        echo "Setting up swap with size: ${swap_size}GB at: $swap_file"

        if [ "$disable_zram" = "true" ]; then
            systemctl disable nvzramconfig
        fi

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
            
            return 0
        else
            echo "Failed to setup swap"
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

    if should_run "gui_disabled" "Would you like to disable the desktop GUI on boot?"; then
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
    if groups $(logname) | grep -q "\bdocker\b"; then
        echo "Docker group already configured, skipping..."
        return 0
    fi

    if should_run "docker_group" "Would you like to add $(logname) to the docker group?"; then
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
    if nvpmodel -q | grep -q "NV Power Mode: MAXN"; then
        echo "Power mode already configured, skipping..."
        return 0
    fi

    if should_run "power_mode" "Would you like to set the power mode to 25W (recommended performance mode)?"; then
        local mode="1"  # Default to 25W mode

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

# Main execution
main() {
    echo "=== Jetson Setup Script ==="
    echo "This script will help configure your Jetson device."
    echo "You will need to reboot once after completion."
    echo

    parse_args "$@"
    check_permissions
    check_dependencies
    detect_nvme_setup
    
    # NVMe Setup
    if should_execute_step "nvme_setup" "Configure NVMe drive"; then
        setup_nvme
    fi

    # Docker Runtime Setup
    if should_execute_step "docker_runtime" "Configure Docker runtime"; then
        setup_docker_runtime
    fi

    # Docker Root Setup
    if should_execute_step "docker_root" "Configure Docker root directory"; then
        setup_docker_root
    fi

    # Swap Setup
    if should_execute_step "swap" "Configure swap space"; then
        setup_swap
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