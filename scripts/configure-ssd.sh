#!/bin/bash
#set -x # Enable debug mode

################################################################################
# JETSON CONTAINER NVME SSD CONFIGURATION SCRIPT
# 
# This script handles NVMe SSD detection and configuration for Jetson devices:
# 1. Detecting if system is installed on NVMe
# 2. Formatting and mounting NVMe storage if needed
# 3. Setting up proper mount points and fstab entries
#
# Usage:
#   ./configure-ssd.sh [OPTIONS]
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
# STORAGE DETECTION AND CONFIGURATION
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
        log ERROR "Unknown storage device: $root_device"
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
    # Prompt user with pre-filled NVMe device
    echo -n "Enter the NVMe device to format [Default: $first_nvme]: "
    read -e -i "$first_nvme" nvme_device
    # Check if the entered device is valid
    if [ ! -b "/dev/$nvme_device" ]; then
        echo "❌ Error: Device /dev/$nvme_device does not exist."
        exit 1
    fi
    # Step 1-3: Ask the user where to mount
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
# HELP AND ARGUMENT PARSING
################################################################################

print_help() {
    echo -e "\n\033[1;34mJetson NVMe SSD Configuration Script\033[0m"
    echo -e "\n\033[1;34mUsage:\033[0m $0 [OPTIONS]"
    echo
    echo -e "\033[1;36mOptions:\033[0m"
    echo -e "  --help             Show this help message and exit."
    echo -e "  --test=<function>  Run a specific function for testing."
    echo
    echo -e "\033[1;36mDescription:\033[0m"
    echo "  This script handles NVMe SSD configuration on Jetson devices."
    echo
    echo -e "\033[1;36mSteps Performed:\033[0m"
    echo "  1️⃣ Detect if Jetson is installed on NVMe or eMMC."
    echo "  2️⃣ If necessary, format and mount the NVMe SSD."
    echo "  3️⃣ Configure proper mount points and fstab entries."
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

################################################################################
# MAIN EXECUTION FLOW
################################################################################

main() {
    parse_args "$@"
    
    # Handle test function execution
    if [[ -n "$TEST_FUNCTION" ]]; then
        if declare -F "$TEST_FUNCTION" > /dev/null; then
            log WARN "Running test for function: $TEST_FUNCTION"
            "$TEST_FUNCTION"
            exit 0
        else
            log ERROR "Function '$TEST_FUNCTION' not found."
            exit 1
        fi
    fi
    
    # Normal execution flow
    print_section "Checking NVMe SSD status"
    if ! check_l4t_installed_on_nvme; then
        if check_nvme_to_be_mounted; then
            mount_nvme
        else
            log WARN "NVMe is either missing or already mounted."
        fi
    else
        log INFO "System is running from NVMe, no additional mounting needed."
    fi
    
    print_section "=== SETUP COMPLETED ==="
}

# Execute main function
main "$@"