#!/bin/bash

# Enable error handling
set -euo pipefail

. scripts/utils.sh

# Load environment variables from .env
load_env

# Default values
interactive_mode="${INTERACTIVE_MODE:-true}"
power_mode_should_run="${POWER_MODE_SHOULD_RUN:-ask}"
power_mode="${POWER_MODE_OPTIONS_MODE:-1}"

# Check for nvpmodel dependency
check_dependencies() {
    if ! is_command_available nvpmodel; then
        echo "Error: Missing required dependency: nvpmodel"
        echo "Please check your Jetson system installation"
        exit 1
    fi
}

get_power_mode() {
    nvpmodel -q | grep "NV Power Mode" | cut -d':' -f2 | xargs
}

# Configure power mode
setup_power_mode() {
    local mode="${power_mode:-1}"
    local current_mode="$(get_power_mode)"
    
    if [ "$current_mode" = "MODE_$mode" ]; then
        echo "Power mode already set to MODE_$mode, skipping..."
        return 0
    fi

    if ask_should_run $power_mode_should_run "Would you like to set the power mode? (May require reboot)"; then
        echo "Setting power mode to mode $mode (this will be applied after reboot)..."
        
        # Use -f flag to suppress the interactive reboot prompt
        if nvpmodel -m "$mode" -f > /dev/null; then
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
    current_mode="$(get_power_mode)"
    echo "Current power mode: $current_mode"
    
    # Configure power mode
    setup_power_mode
    
    # Get updated power mode
    updated_mode="$(get_power_mode)"
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