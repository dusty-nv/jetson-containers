#!/bin/bash

# Enable error handling
set -euo pipefail

. scripts/utils.sh

# Load environment variables from .env
load_env

# Check if NVMe variables are defined
if [ -n "${NVME_SETUP_OPTIONS_MOUNT_POINT+x}" ] && [ -n "${NVME_SETUP_OPTIONS_PARTITION_NAME+x}" ] && [ -n "${NVME_SETUP_OPTIONS_FILESYSTEM+x}" ]; then
    mount_point="$NVME_SETUP_OPTIONS_MOUNT_POINT"
    partition_name="$NVME_SETUP_OPTIONS_PARTITION_NAME"
    filesystem="$NVME_SETUP_OPTIONS_FILESYSTEM"
    nvme_configured=true
else
    nvme_configured=false
fi

# Other variables from .env
docker_root_path="${DOCKER_ROOT_OPTIONS_PATH:-}"
swap_file="${SWAP_OPTIONS_PATH:-}"
disable_zram="${SWAP_OPTIONS_DISABLE_ZRAM:-}"
swap_size="${SWAP_OPTIONS_SIZE:-}"
add_user="${DOCKER_GROUP_OPTIONS_ADD_USER:-}"
power_mode="${POWER_MODE_OPTIONS_MODE:-}"

# Define nvzramconfig service
NVZRAMCONFIG_SERVICE="nvzramconfig"

# Function to check if NVMe is mounted
check_nvme_mount() {
    # Only run if NVMe is configured
    if [ "$nvme_configured" = false ]; then
        echo "NVMe is not configured in environment file."
        return 1
    fi
    
    if mount | grep -q "/dev/$partition_name on $mount_point"; then
        log "✅ NVMe is mounted on $mount_point."
        return 0
    else
        log "❌ NVMe is not mounted on $mount_point."
        return 1
    fi
}

# Function to check Docker data root
check_docker_root() {
    if ! is_docker_installed; then
        return 1
    fi
    
    if [ ! -f "/etc/docker/daemon.json" ]; then
        echo "Docker daemon.json file does not exist."
        return 1
    fi
    
    if [ -z "$docker_root_path" ]; then
        echo "Docker root path is not specified in environment file."
        return 1
    fi
    
    if grep -q '"data-root": "'"$docker_root_path"'"' /etc/docker/daemon.json; then
        echo "Docker data root is set to $docker_root_path."
        return 0
    else
        echo "Docker data root is not set to $docker_root_path."
        return 1
    fi
}

check_swap_file() {
    local swap_file="${1:-$swap_file}"

    # Delegate the “exists and active” test
    if ! check_swap_exists "$swap_file"; then
        # check_swap_exists already printed diagnostics to stderr
        log "❌ Swap is not configured at $swap_file."
        swapon
        return 1
    fi

    # Extract SIZE for the exact swap file, robust to spaces/special chars
    local swap_size_bytes
    swap_size_bytes=""
    while IFS=, read -r name size; do
        # name and size have no surrounding quotes (swapon strips them)
        if [[ "$name" == "$swap_file" ]]; then
            swap_size_bytes="$size"
            break
        fi
    done < <(swapon \
              --bytes \
              --noheadings \
              --raw \
              --output=NAME,SIZE)

    # Validate we found it
    if [[ -z "$swap_size_bytes" ]]; then
        log "❌ Failed to parse swap size for '$swap_file'."
        swapon --show
        return 1
    fi

    # Ensure it's a positive integer
    if ! [[ "$swap_size_bytes" =~ ^[0-9]+$ ]]; then
        log "❌ Unexpected size format for '$swap_file': '$swap_size_bytes'"
        return 1
    fi

    # Compute human‑readable size
    local human_readable_size
    human_readable_size=$(awk -v b="$swap_size_bytes" 'BEGIN { printf "%.2f GB\n", b/1e9 }')

    log "✅ Swap is configured at '$swap_file' with size: $human_readable_size ($swap_size_bytes bytes)."
    return 0
}

# UNIT FILE                                  STATE           VENDOR PRESET
# nvzramconfig.service                       disabled        enabled
check_nvzramconfig_service() {
    # Check if nvzramconfig service is installed
    if systemctl list-unit-files | grep -q "${NVZRAMCONFIG_SERVICE}.service"; then
        # Check if the service is disabled
        if systemctl is-enabled "${NVZRAMCONFIG_SERVICE}.service" &>/dev/null; then
            log "✅ Service '${NVZRAMCONFIG_SERVICE}' is enabled."
            return 1
        else
            log "❌ Service '${NVZRAMCONFIG_SERVICE}' is disabled."
            return 0
        fi
    else
        log "⚠️ Service '${NVZRAMCONFIG_SERVICE}' is not installed."
        return 1
    fi
}

# Function to check zram (nvzramconfig) status
check_zram_status() {
    if systemctl is-enabled nvzramconfig &> /dev/null; then
        log "✅ zram (nvzramconfig) is enabled."
        return 1
    else
        log "❌ zram (nvzramconfig) is disabled."
        return 0
    fi
}

# Function to check swap configuration
check_swap() {
    local swap_check_result=0
    check_swap_file || swap_check_result=1
    check_nvzramconfig_service || swap_check_result=1
    return $swap_check_result
}

# Function to check GUI configuration
check_gui() {
    if systemctl get-default | grep -q "multi-user.target"; then
        log "❌ Desktop GUI is disabled on boot."
        return 0
    else
        log "✅ Desktop GUI is enabled on boot."
        return 1
    fi
}

# Function to check Docker group membership
check_docker_group() {
    if ! is_docker_installed; then
        return 1
    fi
    
    local current_user=$(whoami)
    local result=0
    
    # Check configured user from .env if provided
    if [ -n "$add_user" ]; then
        if groups "$add_user" 2>/dev/null | grep -q "\bdocker\b"; then
            echo "User $add_user is in the docker group."
        else
            echo "User $add_user is not in the docker group."
            result=1
        fi
    fi
    
    # Also check current user if different from the configured user
    if [ -z "$add_user" ] || [ "$current_user" != "$add_user" ]; then
        if groups "$current_user" | grep -q "\bdocker\b"; then
            echo "Current user $current_user is in the docker group."
        else
            echo "Current user $current_user is not in the docker group."
            result=1
        fi
    fi
    
    return $result
}

# Function to check power mode
check_power_mode() {
    local mode
    mode=$(nvpmodel -q | grep "NV Power Mode" | awk -F':' '{gsub(/ /, "", $2); print $2}')
    log "✅ Current power mode: $mode"
    return 0
}

# Function to check if NVMe partition is prepared
check_nvme_partition_prepared() {
    # Only run if NVMe is configured
    if [ "$nvme_configured" = false ]; then
        echo "NVMe is not configured in environment file."
        return 1
    fi
    
    if [ -b "/dev/$partition_name" ] && blkid "/dev/$partition_name" | grep -q "$filesystem"; then
        log "✅ NVMe partition is prepared."
        return 0
    else
        log "❌ NVMe partition is not prepared."
        return 1
    fi
}

# Function to check if NVMe drive is assigned/mounted
check_nvme_drive_assigned() {
    # Only run if NVMe is configured
    if [ "$nvme_configured" = false ]; then
        echo "NVMe is not configured in environment file."
        return 1
    fi
    
    if mount | grep -q "/dev/$partition_name on $mount_point"; then
        log "✅ NVMe drive is already assigned/mounted."
        return 0
    else
        log "❌ NVMe drive is not assigned/mounted."
        return 1
    fi
}

# Function to parse command-line arguments
parse_probe_args() {
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
            --quiet)
                QUIET=true
                shift
                ;;
            --help)
                echo "Usage: probe-system.sh [OPTIONS]"
                echo
                echo "Options:"
                echo "  --tests=<test1,test2,...>    Run specified tests."
                echo "  --help                       Display this help message and exit."
                exit 0
                ;;
            *)
                echo "Unknown parameter passed: $1"
                exit 1
                ;;
        esac
    done
}

# Main function to execute all checks
main() {
    log "=== System Probe Script ==="
    if [ "$nvme_configured" = true ]; then
        log "Assuming NVMe mount point is $mount_point."
    fi
    echo

    parse_probe_args "$@"
    
    log "=== System Probe Script ==="

    if [ ${#TESTS[@]} -eq 0 ]; then
        # Run all checks
        is_docker_installed
        echo
        
        # Only check NVMe if configured
        if [ "$nvme_configured" = true ]; then
            check_nvme_mount
            echo
        fi

        check_docker_runtime
        check_docker_root
        check_swap_file
        check_zram_status
        check_nvzramconfig_service
        check_gui
        check_docker_group
        check_power_mode
    else
        # Run only specified checks
        for test in "${TESTS[@]}"; do
            case $test in
                docker_installed)
                    is_docker_installed
                    echo
                    ;;
                prepare_nvme_partition)
                    check_nvme_partition_prepared
                    ;;
                assign_nvme_drive)
                    check_nvme_drive_assigned
                    ;;
                nvme_mount)
                    check_nvme_mount
                    ;;
                docker_runtime)
                    check_docker_runtime
                    ;;
                docker_root)
                    check_docker_root
                    ;;
                swap_file)
                    check_swap_file
                    ;;
                disable_zram)
                    check_zram_status
                    ;;
                nvzramconfig_service)
                    check_nvzramconfig_service
                    ;;
                gui)
                    check_gui
                    ;;
                docker_group)
                    check_docker_group
                    ;;
                power_mode)
                    check_power_mode
                    ;;
                *)
                    echo "Unknown test: $test"
                    ;;
            esac
        done
    fi
}

# Execute main function
main "$@"

# Execute the probe
exit $?