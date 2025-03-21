#!/bin/bash
#set -x # Enable debug mode

################################################################################
# JETSON CONTAINER DOCKER CONFIGURATION SCRIPT
# 
# This script handles all Docker-related configuration for Jetson devices:
# 1. Formatting and mounting NVMe storage if needed
# 2. Installing Docker with NVIDIA container runtime if needed
# 3. Configuring Docker with NVIDIA runtime as default
# 4. Migrating Docker data directory to NVMe SSD for better performance
# 5. Managing docker group membership for current user
#
# Usage:
#   ./configure-ssd-docker.sh [OPTIONS]
#   
# Options:
#   --help             Show help message
#   --test=<function>  Test a specific function
################################################################################

# Pretty print functions
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

################################################################################
# SECTION 1: STORAGE DETECTION AND CONFIGURATION
################################################################################

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
    log INFO "Setting ownership for $mount_point to ${USER}:${USER}..."
    sudo chown "${USER}:${USER}" "$mount_point"
    ls -la $mount_point
    log INFO "✅ NVMe setup completed successfully!"
}

################################################################################
# SECTION 2: DOCKER INSTALLATION AND CONFIGURATION
################################################################################

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

# Add user to docker group
setup_docker_group() {
    # Check if user is in docker group
    if groups $USER | grep -q '\bdocker\b'; then
        log INFO "✅ User '$USER' is already in the docker group."
    else
        log INFO "Adding user to docker group..."
        sudo usermod -aG docker $USER
        log WARN "⚠️ Please log out and log back in for docker group changes to take effect."
        log WARN "   Alternatively, run: newgrp docker"
    fi
}

# Configure Docker with NVIDIA runtime as default
configure_docker_runtime() {
    if grep -q '"default-runtime": "nvidia"' /etc/docker/daemon.json; then
        log INFO "✅ NVIDIA runtime is already set as default in Docker daemon.json"
    else
        log WARN "❌ NVIDIA runtime is NOT set as default - configuring now..."
        
        # Check if jq is installed
        if ! command -v jq &> /dev/null; then
            log INFO "Installing jq for JSON processing..."
            sudo apt install -y jq
        fi
        
        # Configure NVIDIA runtime as default
        if [ -f "/etc/docker/daemon.json" ]; then
            sudo jq '. + {"default-runtime": "nvidia"}' /etc/docker/daemon.json | \
                sudo tee /etc/docker/daemon.json.tmp > /dev/null && \
                sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json
        else
            echo '{"default-runtime": "nvidia"}' | sudo tee /etc/docker/daemon.json > /dev/null
        fi
        
        # Restart Docker to apply changes
        log INFO "Restarting Docker service to apply changes..."
        sudo systemctl restart docker
        log INFO "✅ NVIDIA runtime is now set as default."
    fi
}

setup_docker() {
    if ! check_docker_is_installed; then
        log ERROR "Docker is not installed"
        exit 1
    fi
    
    # Step 1: Configure Docker runtime
    configure_docker_runtime
    
    # Step 2: Ensure user is in docker group
    setup_docker_group
    
    # Skip data-root migration if L4T is installed on NVMe
    if check_l4t_installed_on_nvme; then
        log WARN "✅ No need to migrate Docker data as your entire system is on NVMe SSD"
        return 0
    fi
    
    # Skip data-root migration if it's already configured
    if grep -q '"data-root"' /etc/docker/daemon.json; then
        log INFO "✅ Docker data-root dir is already set"
        return 0
    fi
    
    # L4T is installed on eMMC and Docker data-root to be migrated to SSD
    log INFO "List all NVMe mount points"
    nvme_mount_points=($(lsblk -nr -o NAME,MOUNTPOINT | awk '/^nvme/ && $2 != "" {print $2}'))
    if [[ ${#nvme_mount_points[@]} -eq 0 ]]; then
        echo "❌ No NVMe mount points found. Exiting."
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
    
    # Migrate Docker data-root directory to SSD
    log INFO "Migrate Docker directory to SSD"
    sudo systemctl stop docker
    log INFO "Moving the existing Docker data directory (/var/lib/docker/)..."
    sudo du -csh /var/lib/docker/
    log WARN "⚠️ This may take time if you were operating on eMMC/SD"
    sudo mkdir -p $selected_mount_point_docker_root_dir && \
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

################################################################################
# SECTION 3: HELP AND ARGUMENT PARSING
################################################################################

print_help() {
    echo -e "\n\033[1;34mJetson Docker Configuration Script\033[0m"
    echo -e "\n\033[1;34mUsage:\033[0m $0 [OPTIONS]"
    echo
    echo -e "\033[1;36mOptions:\033[0m"
    echo -e "  --help             Show this help message and exit."
    echo -e "  --test=<function>  Run a specific function for testing."
    echo -e "  --install-docker   Only install Docker with NVIDIA runtime and exit."
    echo -e "  --mount-only       Only mount NVMe storage without modifying Docker."
    echo -e "  --no-mount         Skip NVMe mounting even if available."
    echo -e "  --no-migrate       Skip Docker data directory migration."
    echo
    echo -e "\033[1;36mDescription:\033[0m"
    echo "  This script configures Docker on Jetson devices with proper NVIDIA container"
    echo "  runtime support and optionally migrates Docker data to NVMe storage."
    echo
    echo -e "\033[1;36mSteps Performed:\033[0m"
    echo "  1️⃣ Detect if Jetson is installed on NVMe or eMMC."
    echo "  2️⃣ If necessary, format and mount the NVMe SSD."
    echo "  3️⃣ Install Docker if it's not already installed."
    echo "  4️⃣ Configure default container runtime to NVIDIA."
    echo "  5️⃣ If using eMMC + NVMe, migrate Docker data-root to SSD."
    echo "  6️⃣ Ensure current user is in the docker group."
    echo
    echo -e "\033[1;36mExamples:\033[0m"
    echo -e "  📌 Run full setup:"
    echo -e "      \033[1;32m$0\033[0m"
    echo
    echo -e "  📌 Only install Docker:"
    echo -e "      \033[1;32m$0 --install-docker\033[0m"
    echo
    echo -e "  📌 Test a specific function:"
    echo -e "      \033[1;32m$0 --test=check_docker_is_installed\033[0m"
    echo
}

parse_args() {
    TEST_FUNCTION=""
    INSTALL_DOCKER_ONLY=false
    MOUNT_ONLY=false
    NO_MOUNT=false
    NO_MIGRATE=false
    
    for arg in "$@"; do
        case "$arg" in
            --test=*)
                TEST_FUNCTION="${arg#*=}"
                ;;
            --install-docker)
                INSTALL_DOCKER_ONLY=true
                ;;
            --mount-only)
                MOUNT_ONLY=true
                ;;
            --no-mount)
                NO_MOUNT=true
                ;;
            --no-migrate)
                NO_MIGRATE=true
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

################################################################################
# SECTION 4: MAIN EXECUTION FLOW
################################################################################

main() {
    parse_args "$@"
    
    # Handle test function execution
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
    
    # Install Docker only and exit if requested
    if [[ "$INSTALL_DOCKER_ONLY" == true ]]; then
        print_section "Installing Docker with NVIDIA runtime"
        if ! check_docker_is_installed; then
            install_docker
            if [ $? -eq 0 ]; then
                configure_docker_runtime
                setup_docker_group
            fi
        else
            log INFO "Docker is already installed. Configuring runtime and group..."
            configure_docker_runtime
            setup_docker_group
        fi
        exit 0
    fi
    
    # Mount NVMe only and exit if requested
    if [[ "$MOUNT_ONLY" == true ]]; then
        print_section "Mounting NVMe storage only"
        if ! check_l4t_installed_on_nvme; then
            if check_nvme_to_be_mounted; then
                mount_nvme
            else
                log WARN "NVMe is either missing or already mounted."
            fi
        else
            log INFO "No need to mount NVMe as system is already running from it."
        fi
        exit 0
    fi
    
    # Normal execution flow if no special flags are supplied
    log INFO "Running full setup..."
    
    # Step 1: Check L4T is installed entirely on NVMe (or eMMC)
    print_section "Step 1: Check L4T is installed entirely on NVMe (or installed on eMMC/SD)"
    check_l4t_installed_on_nvme
    system_on_nvme=$?
    
    # Step 2: Mount NVMe if there is one exists and to be mounted (unless --no-mount)
    if [[ "$NO_MOUNT" != true ]]; then
        print_section "Step 2: Mount NVMe if needed"
        if [ $system_on_nvme -ne 0 ]; then
            log INFO "Checking if NVMe is present and needs mounting"
            if check_nvme_to_be_mounted; then
                mount_nvme
            else
                log WARN "NVMe is either missing or already mounted."
            fi
        else
            log INFO "System is running from NVMe, no additional mounting needed."
        fi
    else
        log INFO "Skipping NVMe mount check (--no-mount option used)"
    fi
    
    # Step 3: Install Docker if not installed
    print_section "Step 3: Install Docker if needed"
    if ! check_docker_is_installed; then
        install_docker
    fi
    
    # Step 4: Configure Docker and migrate if needed
    if [[ "$NO_MIGRATE" != true ]]; then
        print_section "Step 4: Configure Docker runtime and migrate data to SSD if needed"
        setup_docker
    else
        print_section "Step 4: Configure Docker runtime (skipping data migration)"
        configure_docker_runtime
        setup_docker_group
    fi
    
    print_section "=== SETUP COMPLETED ==="
    log INFO "Docker is now configured with NVIDIA runtime as default"
    
    # Remind user to log out and back in if group was added
    if ! groups $USER | grep -q '\bdocker\b'; then
        log WARN "⚠️ IMPORTANT: Log out and back in for docker group changes to take effect"
        log WARN "              Alternatively, run 'newgrp docker' in your current terminal"
    fi
}

# Execute main function
main "$@"
