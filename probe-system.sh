#!/bin/bash

# Function to check if NVMe is mounted
# /dev/nvme0n1 on /mnt type ext4 (rw,relatime)
check_nvme_mount() {
    if mount | grep -q "/dev/nvme0n1"; then
        echo "NVMe is mounted."
        return 0
    else
        echo "NVMe is not mounted."
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

# Function to check swap configuration
check_swap() {
    if swapon --show | grep -q "/mnt/"; then
        echo "Swap is configured at /mnt/."
        return 0
    else
        echo "Swap is not configured at /mnt/16GB.swap."
        return 1
    fi
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
    if groups "$(logname)" | grep -q "\bdocker\b"; then
        echo "User is in the docker group."
        return 0
    else
        echo "User is not in the docker group."
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
            *)
                echo "Unknown parameter passed: $1"
                exit 1
                ;;
        esac
    done
}

# Main function to execute all checks
main() {
    echo "=== System Probe Script ==="
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

        check_swap
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
                swap)
                    check_swap
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