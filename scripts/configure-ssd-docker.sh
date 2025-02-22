#!/bin/bash

#set -x # Enable debug mode

print_section() {
    local message=$1
    echo -e "\n\033[48;5;130m\033[97m >>> $message \033[0m"
}

log() {
    local level=$1
    local message=$2
    case "$level" in
        INFO)  echo -e "\033[36m[INFO]\033[0m $message" ;;      # Cyan text
        WARN)  echo -e "\033[38;5;205m[WARN]\033[0m $message" ;; # Magenta text
        ERROR) echo -e "\033[1;31m[ERROR]\033[0m $message" ;;   # Red text (bold)
        *)     echo "[LOG]  $message" ;;                        # Default log
    esac
}

check_l4t_installed_on_nvme() {
    root_device=$(findmnt -n -o SOURCE /)

    if [[ "$root_device" =~ nvme ]]; then
        log INFO "System is installed on an NVMe SSD ($root_device)."
        return 0
    elif [[ "$root_device" =~ mmcblk ]]; then
        log WARN "System is installed on an eMMC device ($root_device)."
        return 1
    else
        log RED "Unknown storage device: $root_device"
        return 1
    fi
}

check_nvme_should_mount() {
    local nvme_present=false
    local nvme_mounted=false

    # Check if NVMe is detected
    if lsblk -d -o NAME,TYPE | grep -q "nvme"; then
        nvme_present=true
        log INFO "NVMe device is present"
    fi

    # Check if NVMe is mounted
    if lsblk -o NAME,MOUNTPOINT | grep -q "nvme.*\/"; then
        nvme_mounted=true
        log INFO "NVMe device is already mounted"
    fi

    # Determine return status
    if [ "$nvme_present" = true ] && [ "$nvme_mounted" = false ]; then
        echo "✅ NVMe is present and should be mounted."
        return 0  # SUCCESS: NVMe present but NOT mounted (should be mounted)
    else
        echo "❌ NVMe does not need mounting."
        return 1  # FAILURE: Either NVMe is missing or already mounted
    fi
}

mount_nvme() {
    # Step 2: List available NVMe devices
    log INFO "Available NVMe devices:"
    lsblk -d -o NAME,SIZE,TYPE | grep "nvme"

    # Step 3: Detect the first NVMe device automatically
    first_nvme=$(lsblk -d -o NAME,TYPE | grep "nvme" | awk '{print $1}' | head -n1)

    # Prompt user with pre-filled NVMe device (default: first detected NVMe)
    echo -n "Enter the NVMe device to format [Default: $first_nvme]: "
    read -e -i "$first_nvme" nvme_device  # Pre-fill input with the first detected NVMe

    # Check if the entered device is valid
    if [ ! -b "/dev/$nvme_device" ]; then
        echo "❌ Error: Device /dev/$nvme_device does not exist."
        exit 1
    fi

    # Step 4: Ask the user where to mount (default: /mnt)
    echo -n "Enter the mount point [default: /mnt]                 : "
    read -e -i "/mnt" mount_point
    mount_point=${mount_point:-/mnt}

    # Step 5: Format the NVMe device
    log WARN "⚠️ WARNING: This will format /dev/$nvme_device as EXT4!"
    read -p "Are you sure you want to proceed? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        log ERROR "❌ Aborting formatting."
        exit 1
    fi

    log INFO "Formatting /dev/$nvme_device..."
    sudo mkfs.ext4 "/dev/$nvme_device"

    # Step 6: Create mount directory
    log INFO "Creating mount directory at $mount_point..."
    sudo mkdir -p "$mount_point"

    # Step 7: Mount the NVMe device
    log INFO "Mounting /dev/$nvme_device to $mount_point..."
    sudo mount "/dev/$nvme_device" "$mount_point"

    # Step 8: Display filesystem info
    log INFO "Updated Filesystem Info:"
    lsblk -f

    # Step 9: Get UUID of the new filesystem
    uuid=$(blkid -s UUID -o value "/dev/$nvme_device")

    # Step 10: Append fstab entry
    fstab_entry="UUID=$uuid $mount_point ext4 defaults 0 2"

    log INFO "Adding the following entry to /etc/fstab:"
    echo "$fstab_entry"

    # Open fstab for editing
    sudo bash -c "echo '$fstab_entry' >> /etc/fstab"

    # Step 11: Change ownership to current user
    log IFNO "Setting ownership for $mount_point to ${USER}:${USER}..."
    sudo chown "${USER}:${USER}" "$mount_point"

    log INFO "✅ NVMe setup completed successfully!"
}

check_docker_is_installed() {
    if command -v docker &> /dev/null; then
        log INFO "✅ Docker is installed."
        return 0
    else
        log WARN "❌ Docker is NOT installed."
        return 1
    fi
}

install_docker() {
    log INFO "⚠️  Docker and NVIDIA container runtime will be installed."
    log INFO "   This process requires internet access and may take a few minutes."
    echo

    # Ask for user confirmation
    read -p "Are you sure you want to install Docker? (y/N): " confirm
    confirm=${confirm,,}  # Convert to lowercase

    if [[ "$confirm" != "y" && "$confirm" != "yes" ]]; then
        echo "❌ Installation aborted by user."
        return 1
    fi

    # Proceed with installation
    log INFO "✅ Proceeding with Docker installation..."
    sudo apt update
    sudo apt install -y nvidia-container curl
    curl https://get.docker.com | sh && sudo systemctl --now enable docker
    sudo nvidia-ctk runtime configure --runtime=docker

    log INFO "✅ Docker and NVIDIA runtime installation complete!"
}

setup_docker() {

    if ! check_docker_is_installed; then
        log ERROR "Docker is not installed"
        exit 1
    fi

    log INFO "Going to restart docker service..."
    sudo systemctl restart docker
    sudo usermod -aG docker $USER
    log WARN "⚠️ Please log out and log back in for Docker group changes to take effect."

    sudo apt install -y jq
    sudo jq '. + {"default-runtime": "nvidia"}' /etc/docker/daemon.json | \
        sudo tee /etc/docker/daemon.json.tmp && \
        sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json

    if check_l4t_installed_on_nvme; then
        log WARN "No need to data-root for Docker config"
    else
        log INFO "Migrate Docker directory to SSD"
        sudo systemctl stop docker

        log INFO "Moving the existing Docker directory..."
        sudo du -csh /var/lib/docker/ && \
            sudo mkdir /ssd/docker && \
            sudo rsync -axPS /var/lib/docker/ /ssd/docker/ && \
            sudo du -csh  /ssd/docker/

        log INFO "List all NVMe mount points"
        nvme_mount_points=($(awk '/^\/dev\/nvme/ {print $2}' /etc/fstab))
        if [[ ${#nvme_mount_points[@]} -eq 0 ]]; then
            echo "❌ No NVMe mount points found in /etc/fstab. Exiting."
            exit 1
        fi
        # List all NVMe mount points
        for mp in "${nvme_mount_points[@]}"; do
            echo "Mounted NVMe: $mp"
        done

        # Set default to the first NVMe mount point
        default_mount_point_docker_root_dir="${nvme_mount_points[0]}/docker"
        echo -n "Enter the Docker data-root dir [Default: $default_mount_point_docker_root_dir]: "
        read -e -i "$default_mount_point_docker_root_dir" selected_mount_point_docker_root_dir

        # Insert Docker data-root entry in daemon.json safely
        sudo jq --arg dataroot "$selected_mount_point_docker_root_dir" \
            '. + {"data-root": $dataroot}' /etc/docker/daemon.json | \
            sudo tee /etc/docker/daemon.json.tmp > /dev/null

        # Validate JSON before replacing daemon.json
        if sudo jq empty /etc/docker/daemon.json.tmp >/dev/null 2>&1; then
            sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json
            echo "✅ Docker data-root updated to: $selected_mount_point_docker_root_dir"
        else
            echo "❌ JSON validation failed. Aborting update."
            rm /etc/docker/daemon.json.tmp
        fi

        # Rename the old Docker data directory
        sudo mv /var/lib/docker /var/lib/docker.old

        # Restart the docker daemon
        sudo systemctl daemon-reload && \
            sudo systemctl restart docker && \
            sudo journalctl -u docker
    fi
}

parse_args() {
    TEST_FUNCTION=""

    for arg in "$@"; do
        case "$arg" in
            --test=*)
                TEST_FUNCTION="${arg#*=}"
                ;;
            --help)
                print_help
                exit 0
                ;;
            *)
                log ERROR "Unknown parameter: $arg"
                exit 1
                ;;
        esac
    done
}

main() {

    parse_args "$@"

    if [[ -n "$TEST_FUNCTION" ]]; then
        # Check if the function exists before calling
        if declare -F "$TEST_FUNCTION" > /dev/null; then
            log WARN "Running test for function: $TEST_FUNCTION"
            "$TEST_FUNCTION"  # Call the function dynamically
            exit 0
        else
            log ERROR "Function '$TEST_FUNCTION' not found."
            exit 1
        fi
    fi

    # Normal execution flow if no --test flag is supplied
    log INFO "Running full setup..."

    print_section "Check and install Docker (if not installed)"
    if command -v docker &> /dev/null; then
        log INFO "Docker is already installed."
    else
        log WARN "Docker is NOT yet installed."
        sudo apt update
        sudo apt install -y nvidia-container curl
        curl https://get.docker.com | sh && sudo systemctl --now enable docker
        sudo nvidia-ctk runtime configure --runtime=docker
    fi

    # Step 0:
    if check_l4t_installed_on_nvme; then
        log INFO "No need to mount SSD, no need to set Docker data-root dir."
    else
        log INFO "Checking if NVMe is present and to be mounted"

        # Step 1: Check if NVMe needs mounting
        if check_nvme_should_mount; then
            mount_nvme
        else
            log NVME "NVMe is either missing or already mounted. Exiting."
        fi
    fi

    # Check if Docker is installed and install Docker if not installed
    if ! check_docker_is_installed; then
        install_docker
    fi

    # If eMMC + NVMe, migrate Docker directory to SSD
    setup_docker

}

# Execute main function
main "$@"