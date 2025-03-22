#!/bin/bash

# Enable error handling
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load environment variables from .env
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
elif [ -f "${SCRIPT_DIR}/../.env" ]; then
    set -a
    source "${SCRIPT_DIR}/../.env"
    set +a
fi

# Default values
interactive_mode="${INTERACTIVE_MODE:-true}"

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

# Check if script is run with sudo
check_permissions() {
    if [ "$EUID" -ne 0 ]; then 
        echo "Please run as root (with sudo)"
        exit 1
    fi
}

# Parse command line arguments
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
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Configure Jetson system settings"
                echo ""
                echo "Options:"
                echo "  --tests=TESTS     Comma-separated list of tests to run"
                echo "  --help            Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown parameter passed: $1"
                exit 1
                ;;
        esac
    done
}

# Execute a configuration script if conditions are met
execute_config_script() {
    local script="$1"
    local prompt="$2"
    local var_name="$3"
    local extra_args="${4:-}"
    
    # Check if script exists and is executable
    if [ ! -x "$script" ]; then
        echo "Error: Configuration script $script not found or not executable"
        return 1
    fi
    
    # Determine if we should run this script
    local should_run="${!var_name:-ask}"
    if [ "$execution_mode" = "all" ] || \
       [ "$should_run" = "yes" ] || \
       ([ "$should_run" = "ask" ] && [ "$interactive_mode" = "true" ] && ask_yes_no "$prompt"); then
        echo "=== System Probe Script === (Before configuration)"
        "${SCRIPT_DIR}/probe-system.sh"
        echo "Running $(basename "$script")..."
        "$script" $extra_args
        echo "=== System Probe Script === (After configuration)"
        "${SCRIPT_DIR}/probe-system.sh"
    fi
}

# Main execution
main() {
    check_permissions
    parse_args "$@"
    
    if [ ${#TESTS[@]} -eq 0 ]; then
        ask_execution_mode
    else
        execution_mode="specific"
    fi
    
    echo "=== Jetson Setup Script ==="
    echo "This script will help configure your Jetson device."
    echo "You will need to reboot once after completion."
    echo
    
    # Initial system probe
    echo "Probing status before working"
    echo "============================="
    "${SCRIPT_DIR}/probe-system.sh"
    echo "============================="
    
    # Prepare extra arguments for scripts where needed
    local swap_args=""
    if [ "${SWAP_OPTIONS_DISABLE_ZRAM:-false}" = "true" ]; then
        echo "Including --disable-zram flag from .env settings"
        swap_args="--disable-zram"
    fi
    
    # Execute configuration scripts
    execute_config_script "${SCRIPT_DIR}/configure-ssd.sh" "Configure NVMe SSD storage?" "SSD_SETUP_SHOULD_RUN"
    execute_config_script "${SCRIPT_DIR}/configure-docker.sh" "Configure Docker?" "DOCKER_SETUP_SHOULD_RUN"
    execute_config_script "${SCRIPT_DIR}/configure-swap.sh" "Configure swap?" "SWAP_SHOULD_RUN" "$swap_args"
    execute_config_script "${SCRIPT_DIR}/configure-system-gui.sh" "Configure desktop GUI?" "GUI_DISABLED_SHOULD_RUN"
    execute_config_script "${SCRIPT_DIR}/configure-power-mode.sh" "Configure power mode?" "POWER_MODE_SHOULD_RUN"
    
    echo
    echo "Configuration complete!"
    echo "============================="
    "${SCRIPT_DIR}/probe-system.sh"
    echo "============================="
    
    echo "Please reboot your system for all changes to take effect."
    if [ "$interactive_mode" = "true" ] && ask_yes_no "Would you like to reboot now?"; then
        reboot
    fi
}

# Run main function with all passed arguments
main "$@"