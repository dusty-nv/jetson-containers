#!/bin/bash

# Load environment variables from .env
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
else
    echo "Environment file .env not found."
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

# Global background flag
BG=false

# Function to check if NVMe is mounted
check_nvme_mount() {
    if mount | grep -q "/dev/$partition_name on $mount_point"; then
        echo "NVMe is mounted on $mount_point."
        return 0
    else
        echo "NVMe is not mounted on $mount_point."
        return 1
    fi
}

# Function to check Docker runtime configuration
check_docker_runtime() {
    if grep -q '"default-runtime": "nvidia"' /etc/docker/daemon.json; then
        echo "Docker runtime 'nvidia' is set as default."
        return 0
    else
        echo "Docker runtime 'nvidia' is not set as default."
        return 1
    fi
}

# Function to check Docker data root
check_docker_root() {
    if grep -q "\"data-root\": \"$docker_root_path\"" /etc/docker/daemon.json; then
        echo "Docker data root is set to $docker_root_path."
        return 0
    else
        echo "Docker data root is not set to $docker_root_path."
        return 1
    fi
}

check_swap_file() {
    if swapon --show | grep -q "$swap_file"; then
        local swap_size_bytes
        swap_size_bytes=$(swapon --show=SIZE --bytes "$swap_file" | tail -n1)
        echo "Swap is configured at $swap_file with size: $swap_size_bytes bytes."
        return 0
    else
        echo "Swap is not configured at $swap_file."
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
            echo "Service '${NVZRAMCONFIG_SERVICE}' is enabled."
            return 1
        else
            echo "Service '${NVZRAMCONFIG_SERVICE}' is disabled."
            return 0
        fi
    else
        echo "Service '${NVZRAMCONFIG_SERVICE}' is not installed."
        return 1
    fi
}

# Function to check zram (nvzramconfig) status
check_zram() {
    if systemctl is-enabled nvzramconfig &> /dev/null; then
        echo "zram (nvzramconfig) is enabled."
        return 1
    else
        echo "zram (nvzramconfig) is disabled."
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
        echo "Desktop GUI is disabled on boot."
        return 0
    else
        echo "Desktop GUI is enabled on boot."
        return 1
    fi
}

# Function to check Docker group membership
check_docker_group() {
    if groups "$add_user" | grep -q "\bdocker\b"; then
        echo "User $add_user is in the docker group."
        return 0
    else
        echo "User $add_user is not in the docker group."
        return 1
    fi
}

# Function to check power mode
check_power_mode() {
    local mode
    mode=$(nvpmodel -q | grep "NV Power Mode" | awk -F':' '{gsub(/ /, "", $2); print $2}')
    echo "Current power mode: $mode"
    return 0
}

# Function to check if NVMe partition is prepared
check_nvme_partition_prepared() {
    if [ -b "/dev/$partition_name" ] && blkid "/dev/$partition_name" | grep -q "$filesystem"; then
        echo "NVMe partition is prepared."
        return 0
    else
        echo "NVMe partition is not prepared."
        return 1
    fi
}

# Function to check if NVMe drive is assigned/mounted
check_nvme_drive_assigned() {
    if mount | grep -q "/dev/$partition_name on $mount_point"; then
        echo "NVMe drive is already assigned/mounted."
        return 0
    else
        echo "NVMe drive is not assigned/mounted."
        return 1
    fi
}

# Function to display help information
print_help() {
    echo "Usage: probe-system.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  --tests=<test1,test2,...>    Run specified tests."
    echo "  --bg                         Run in background mode."
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
            --bg)
                BG=true
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

# Main function to execute checks
main() {
    # Print header only if not running in background mode
    if [ "$BG" = false ]; then
        echo "=== System Probe Script ==="
        echo "Assuming NVMe mount point is $mount_point."
        echo
    fi

    parse_probe_args "$@"

    # If no tests provided, run all checks normally
    if [ ${#TESTS[@]} -eq 0 ]; then
        check_nvme_mount
        echo

        check_docker_runtime
        echo

        check_docker_root
        echo

        check_swap_file
        echo

        check_zram
        echo

        check_nvzramconfig_service
        echo

        check_gui
        echo

        check_docker_group
        echo

        check_power_mode
        echo
    else
        # Run only specified tests and aggregate exit codes
        rc=0
        for test in "${TESTS[@]}"; do
            case $test in
                prepare_nvme_partition)
                    check_nvme_partition_prepared || rc=1
                    echo
                    ;;
                assign_nvme_drive)
                    check_nvme_drive_assigned || rc=1
                    echo
                    ;;
                nvme_mount)
                    check_nvme_mount || rc=1
                    echo
                    ;;
                docker_runtime)
                    check_docker_runtime || rc=1
                    echo
                    ;;
                docker_root)
                    check_docker_root || rc=1
                    echo
                    ;;
                swap_file)
                    check_swap_file || rc=1
                    echo
                    ;;
                disable_zram)
                    check_zram || rc=1
                    echo
                    ;;
                nvzramconfig_service)
                    check_nvzramconfig_service || rc=1
                    echo
                    ;;
                gui)
                    check_gui || rc=1
                    echo
                    ;;
                docker_group)
                    check_docker_group || rc=1
                    echo
                    ;;
                power_mode)
                    check_power_mode || rc=1
                    echo
                    ;;
                *)
                    echo "Unknown test: $test"
                    rc=1
                    ;;
            esac
        done
        # If background flag set, exit silently with aggregated result
        if [ "$BG" = true ]; then
            exit $rc
        fi
    fi
}

# Execute main function and exit with its code
main "$@"
exit $?