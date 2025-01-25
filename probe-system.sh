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
    if grep -q '"data-root": "/mnt/docker"' /etc/docker/daemon.json; then
        echo "Docker data root is set to /mnt/docker."
        return 0
    else
        echo "Docker data root is not set to /mnt/docker."
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

# Function to probe system status
probe_system() {
    # Example checks
    if ! command -v docker &> /dev/null; then
        echo "Docker is not installed."
        return 1
    fi
    # ...additional probes...
    return 0
}

# Main function to execute all checks
main() {
    echo "=== System Probe Script ==="
    echo "Assuming NVMe mount point is $mount_point."
    echo

    parse_probe_args "$@"

    if [ ${#TESTS[@]} -eq 0 ]; then
        # Run all checks
        check_nvme_mount
        echo

        check_docker_runtime
        echo

        check_docker_root
        echo

        check_swap_file
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
        # Run only specified checks
        for test in "${TESTS[@]}"; do
            case $test in
                nvme_mount)
                    check_nvme_mount
                    echo
                    ;;
                docker_runtime)
                    check_docker_runtime
                    echo
                    ;;
                docker_root)
                    check_docker_root
                    echo
                    ;;
                swap_file)
                    check_swap_file
                    echo
                    ;;
                nvzramconfig_service)
                    check_nvzramconfig_service
                    echo
                    ;;
                gui)
                    check_gui
                    echo
                    ;;
                docker_group)
                    check_docker_group
                    echo
                    ;;
                power_mode)
                    check_power_mode
                    echo
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
probe_system
exit $?