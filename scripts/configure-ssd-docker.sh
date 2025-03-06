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

check_nvme_to_be_mounted() {
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

add_nvme_to_fstab() {
    local nvme_device="$1"    # e.g., /dev/nvme0n1
    local mount_point="$2"    # e.g., /mnt
    local filesystem="ext4"

    # Refresh blkid cache to prevent missing UUID issue
    sudo blkid -c /dev/null > /dev/null

    # Get UUID of the NVMe device
    blkid -s UUID -o value "$nvme_device"
    local uuid
    uuid=$(blkid -s UUID -o value "$nvme_device")
    log INFO "------> UUID: $uuid"

    # Ensure the UUID was found
    if [[ -z "$uuid" ]]; then
        echo "❌ Failed to retrieve UUID for $nvme_device. Is it formatted?"
        return 1
    fi

    # Check if the entry already exists in /etc/fstab
    if grep -q "$uuid" /etc/fstab; then
        echo "✅ UUID $uuid already exists in /etc/fstab. No changes needed."
        return 0
    fi

    # Append entry to /etc/fstab
    echo "UUID=$uuid $mount_point $filesystem defaults 0 2" | sudo tee -a /etc/fstab

    # Apply changes
    sudo mount -a

    echo "✅ Successfully added $nvme_device to /etc/fstab and mounted it."
}

mount_nvme() {
    # Step 1-1: List available NVMe devices
    log INFO "Available NVMe devices:"
    lsblk -d -o NAME,SIZE,TYPE | grep "nvme"

    # Step 1-2: Detect the first NVMe device automatically
    first_nvme=$(lsblk -d -o NAME,TYPE | grep "nvme" | awk '{print $1}' | head -n1)

    # Prompt user with pre-filled NVMe device (default: first detected NVMe)
    echo -n "Enter the NVMe device to format [Default: $first_nvme]: "
    read -e -i "$first_nvme" nvme_device  # Pre-fill input with the first detected NVMe

    # Check if the entered device is valid
    if [ ! -b "/dev/$nvme_device" ]; then
        echo "❌ Error: Device /dev/$nvme_device does not exist."
        exit 1
    fi

    # Step 1-3: Ask the user where to mount (default: /mnt)
    echo -n "Enter the mount point           [Default: /mnt]   : "
    read -e -i "/mnt" mount_point
    mount_point=${mount_point:-/mnt}

    # Step 1-4: Format the NVMe device
    log WARN "⚠️ WARNING: This will format /dev/$nvme_device as EXT4!"
    read -p "Are you sure you want to proceed? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        log ERROR "❌ Aborting formatting."
        exit 1
    fi

    log INFO "Formatting /dev/$nvme_device..."
    sudo mkfs.ext4 "/dev/$nvme_device"

    # Step 1-5: Create mount directory
    log INFO "Creating mount directory at $mount_point..."
    sudo mkdir -p "$mount_point"

    # Step 1-6: Mount the NVMe device
    log INFO "Mounting /dev/$nvme_device to $mount_point..."
    sudo mount "/dev/$nvme_device" "$mount_point"

    # Step 1-7: Display filesystem info
    log INFO "Updated Filesystem Info:"
    lsblk -f

    # Step 1-8: Get the UUID and add that to /etc/fstab
    add_nvme_to_fstab "/dev/$nvme_device" "$mount_point"
    log INFO "Showing the current /etc/fstab"
    cat /etc/fstab

    # Step 1-9: Change ownership to current user
    log IFNO "Setting ownership for $mount_point to ${USER}:${USER}..."
    sudo chown "${USER}:${USER}" "$mount_point"
    ls -la $mount_point

    log INFO "✅ NVMe setup completed successfully!"
}

check_docker_is_installed() {
    if command -v docker &> /dev/null; then
        log INFO "✅ Docker is already installed."
        return 0
    else
        log WARN "❌ Docker is NOT installed."
        return 1
    fi
}

install_docker() {
    log INFO "⚠️ Docker and NVIDIA container runtime will be installed."
    log INFO "⚠️ This process requires internet access and may take a few minutes."
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
    cd  /tmp/
    if [ -d "install-docker" ]; then
        rm -rf install-docker
    fi
    git clone https://github.com/jetsonhacks/install-docker.git
    cd install-docker
    bash ./install_nvidia_docker.sh

    log INFO "✅ Docker and NVIDIA runtime installation complete!"
}

setup_docker() {

    if ! check_docker_is_installed; then
        log ERROR "Docker is not installed"
        exit 1
    fi

    # Step 3-1: Add default-runtime (if needed)
    if grep -q '"default-runtime": "nvidia"' /etc/docker/daemon.json; then
        log INFO "✅ NVIDIA runtime is set as default"
    else
        log WARN "❌ NVIDIA runtime is NOT set"
        log INFO "Going to restart docker service..."
        sudo systemctl restart docker
        sudo usermod -aG docker $USER
        log WARN "⚠️ Please log out and log back in for Docker group changes to take effect."

        sudo apt install -y jq
        sudo jq '. + {"default-runtime": "nvidia"}' /etc/docker/daemon.json | \
            sudo tee /etc/docker/daemon.json.tmp && \
            sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json
    fi

    if check_l4t_installed_on_nvme; then
        log WARN "✅ No need to migrate Docker data as you entire system is on NVMe SSD"
        exit 1
    fi

    if grep -q '"data-root"' /etc/docker/daemon.json; then
        log INFO "✅ Docker data-root dir is already set"
        exit 1
    fi

    # L4T is installed on eMMC and Docker data-root to be migrated to SSD
    log INFO "List all NVMe mount points"
    nvme_mount_points=($(lsblk -nr -o NAME,MOUNTPOINT | awk '/^nvme/ && $2 != "" {print $2}'))
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

    # Step 3-2: Migrate Docker data-root directory to SSD (if needed)
    log INFO "Migrate Docker directory to SSD"
    sudo systemctl stop docker

    log INFO "Moving the existing Docker data directory (/var/lib/docker/)..."
    sudo du -csh /var/lib/docker/
    log WARN "⚠️ This may take time if you were operating on eMMC/SD"
    sudo mkdir $selected_mount_point_docker_root_dir && \
        sudo rsync -axPS /var/lib/docker/ $selected_mount_point_docker_root_dir && \
        sudo du -csh  $selected_mount_point_docker_root_dir

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
        sudo systemctl restart docker

}

print_help() {
    echo -e "\n\033[1;34mUsage:\033[0m $0 [OPTIONS]"
    echo
    echo -e "\033[1;36mOptions:\033[0m"
    echo -e "  --help             Show this help message and exit."
    echo -e "  --test=<function>  Run a specific function for testing."
    echo
    echo -e "\033[1;36mDescription:\033[0m"
    echo "  This script configures an NVMe SSD as storage for Docker on a Jetson device."
    echo "  It will detect if Jetson is running from eMMC and migrate Docker data if needed."
    echo
    echo -e "\033[1;36mSteps Performed:\033[0m"
    echo "  1️⃣ Detect if Jetson is installed on NVMe or eMMC."
    echo "  2️⃣ If necessary, format and mount the NVMe SSD."
    echo "  3️⃣ Install Docker if it's not already installed."
    echo "  4️⃣ If using eMMC + NVMe, migrate Docker data-root to SSD."
    echo
    echo -e "\033[1;36mExamples:\033[0m"
    echo -e "  📌 Run full setup:"
    echo -e "      \033[1;32m$0\033[0m"
    echo
    echo -e "  📌 Test a specific function (e.g., check if Docker is installed):"
    echo -e "      \033[1;32m$0 --test=check_docker_is_installed\033[0m"
    echo
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

    # Step 1: Check L4T is installed entirely on NVMe (or eMMC)
    print_section "Step 1: Check L4T is installed entirely on NVMe (or installed on eMMC/SD)"
    if check_l4t_installed_on_nvme; then
        log INFO "No need to mount SSD, no need to set Docker data-root dir. (Skipping Step 2)"
    else
        log INFO "Checking if NVMe is present and to be mounted"
        # Step 2: Mount NVMe if there is one exists and to be mounted
        print_section "Step 2: Mount NVMe if there is one exists and to be mounted"
        if check_nvme_to_be_mounted; then
            mount_nvme
        else
            log WARN "NVMe is either missing or already mounted."
        fi
    fi

    # Step 3: Install Docker if not installed
    print_section "Step 3: Install Docker if not installed"
    if ! check_docker_is_installed; then
        install_docker
    fi

    # Step 4: If eMMC + NVMe, migrate Docker directory to SSD
    print_section "Step 4: If eMMC + NVMe, migrate Docker directory to SSD"
    setup_docker

    print_section "=== COMPLETED ==="
}

# Execute main function
main "$@"
