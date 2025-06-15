#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(dirname "$(dirname "$(readlink -fm "$0")")")"

# Global flag for quiet mode
QUIET=false

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

mount_point="$NVME_SETUP_OPTIONS_MOUNT_POINT"
swap_file="$SWAP_OPTIONS_PATH"
partition_name="$NVME_SETUP_OPTIONS_PARTITION_NAME"
filesystem="$NVME_SETUP_OPTIONS_FILESYSTEM"
docker_root_path="$DOCKER_ROOT_OPTIONS_PATH"
disable_zram="$SWAP_OPTIONS_DISABLE_ZRAM"
swap_size="$SWAP_OPTIONS_SIZE"
add_user="$DOCKER_GROUP_OPTIONS_ADD_USER"
power_mode="$POWER_MODE_OPTIONS_MODE"

# Define nvzramconfig service
NVZRAMCONFIG_SERVICE="nvzramconfig"


# Unified logging function
log() {
    if [ "$QUIET" = false ]; then
        echo "$@"
    fi
}

# Function to check if NVMe is mounted
check_nvme_mount() {
    log "ℹ️  Assuming NVMe mount point is $mount_point."

    if mount | grep -q "/dev/$partition_name on $mount_point"; then
        log "✅ NVMe is mounted on $mount_point."
        return 0
    else
        log "❌ NVMe is not mounted on $mount_point."
        return 1
    fi
}

# Function to check Docker runtime configuration
check_docker_runtime() {
    if grep -q '"default-runtime": "nvidia"' /etc/docker/daemon.json; then
        log "✅ Docker runtime 'nvidia' is set as default."
        return 0
    else
        log "❌ Docker runtime 'nvidia' is not set as default."
        return 1
    fi
}

# Function to check Docker data root
check_docker_root() {
    if grep -q '"data-root": "/mnt/docker"' /etc/docker/daemon.json; then
        log "✅ Docker data root is set to /mnt/docker."
        return 0
    else
        log "❌ Docker data root is not set to /mnt/docker."
        return 1
    fi
}

check_swap_file() {
    if swapon --show | grep -q "$swap_file"; then
        local swap_size_bytes
        swap_size_bytes=$(swapon --show=SIZE --bytes "$swap_file" | tail -n1)
        human_readable_size=$(awk -v b="${swap_size_bytes}" 'BEGIN { printf "%.2f GB\n", b/1e9 }')
        log "✅ Swap is configured at $swap_file with size: $human_readable_size ($swap_size_bytes bytes)."
        return 0
    else
        log "❌ Swap is not configured at $swap_file."
        swapon
        return 1
    fi
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
check_zram() {
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
    return 1 && check_swap_file && check_nvzramconfig_service
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
    if groups "$add_user" | grep -q "\bdocker\b"; then
        log "✅ User $add_user is in the docker group."
        return 0
    else
        log "❌ User $add_user is not in the docker group."
        return 1
    fi
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
    if mount | grep -q "/dev/$partition_name on $mount_point"; then
        log "✅ NVMe drive is already assigned/mounted."
        return 0
    else
        log "❌ NVMe drive is not assigned/mounted."
        return 1
    fi
}

# Function to display help information
print_help() {
    echo "Usage: probe-system.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  --tests=<test1,test2,...>    Run specified tests."
    echo "  --help                       Display this help message and exit."
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
                print_help
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
    parse_probe_args "$@"
    
    log "=== System Probe Script ==="

    if [ ${#TESTS[@]} -eq 0 ]; then
        # Run all checks
        check_nvme_mount
        check_docker_runtime
        check_docker_root
        check_swap_file
        check_zram
        check_nvzramconfig_service
        check_gui
        check_docker_group
        check_power_mode
    else
        # Run only specified checks
        for test in "${TESTS[@]}"; do
            case $test in
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
                    check_zram
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