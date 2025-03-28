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
power_mode_should_run="${POWER_MODE_SHOULD_RUN:-ask}"
power_mode="${POWER_MODE_OPTIONS_MODE:-1}"

# Check if script is run with sudo
check_permissions() {
    if [ "$EUID" -ne 0 ]; then 
        echo "Please run as root (with sudo)"
        exit 1
    fi
}

# Check for nvpmodel dependency
check_dependencies() {
    if ! command -v nvpmodel &> /dev/null; then
        echo "Error: Missing required dependency: nvpmodel"
        echo "Please check your Jetson system installation"
        exit 1
    fi
}

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

# Configure power mode
setup_power_mode() {
    local mode="$power_mode"
    local current_mode=$(nvpmodel -q | grep "NV Power Mode" | cut -d':' -f2 | xargs)
    
    if [ "$current_mode" = "MODE_$mode" ]; then
        echo "Power mode already set to MODE_$mode, skipping..."
        return 0
    fi

    if [ "$power_mode_should_run" = "yes" ] || ([ "$power_mode_should_run" = "ask" ] && [ "$interactive_mode" = "true" ] && ask_yes_no "Would you like to set the power mode? (May require reboot)"); then
        echo "Setting power mode to mode $mode (this will be applied after reboot)..."
        
        # Use -f flag to suppress the interactive reboot prompt
        if nvpmodel -m "$mode" -f; then
            echo "Power mode change scheduled. A reboot will be required to apply this change."
            return 0
        else
            echo "Failed to set power mode"
            return 1
        fi
    fi
    return 0
}

# Parse command line arguments
parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --mode=*)
                power_mode="${1#*=}"
                power_mode_should_run="yes"
                shift
                ;;
            --help)
                echo "Usage: $0 [--mode=<0-3>]"
                echo "Configure Jetson power mode"
                echo ""
                echo "Options:"
                echo "  --mode=N    Set power mode to N (0-3, depending on Jetson model)"
                echo "  --help      Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown parameter passed: $1"
                exit 1
                ;;
        esac
    done
}

# Main function
main() {
    check_permissions
    check_dependencies
    parse_args "$@"
    
    echo "=== Jetson Power Mode Configuration ==="
    
    # Get current power mode
    current_mode=$(nvpmodel -q | grep "NV Power Mode" | cut -d':' -f2 | xargs)
    echo "Current power mode: $current_mode"
    
    # Configure power mode
    setup_power_mode
    
    # Get updated power mode
    updated_mode=$(nvpmodel -q | grep "NV Power Mode" | cut -d':' -f2 | xargs)
    echo "Updated power mode: $updated_mode"
    
    echo "Power mode configuration complete."
    
    if [ "$current_mode" != "$updated_mode" ]; then
        echo "A reboot is required for the power mode change to take effect."
        if [ "$interactive_mode" = "true" ] && ask_yes_no "Would you like to reboot now?"; then
            reboot
        fi
    fi
}

# Run main function with arguments
main "$@"