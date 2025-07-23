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
gui_enabled="${GUI_ENABLED:-ask}"

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

# Function to check if system uses systemd
check_systemd() {
    if ! command -v systemctl &> /dev/null; then
        echo "Error: This script requires systemd"
        exit 1
    fi
}

# Function to get current GUI state
get_gui_state() {
    if systemctl get-default | grep -q "graphical.target"; then
        echo "enabled"
    else
        echo "disabled"
    fi
}

# Function to enable GUI
enable_gui() {
    echo "Enabling GUI on boot..."
    sudo systemctl set-default graphical.target
    if [ "$(get_gui_state)" = "enabled" ]; then
        echo "✅ GUI has been enabled on boot"
        return 0
    else
        echo "❌ Failed to enable GUI"
        return 1
    fi
}

# Function to disable GUI
disable_gui() {
    echo "Disabling GUI on boot..."
    sudo systemctl set-default multi-user.target
    if [ "$(get_gui_state)" = "disabled" ]; then
        echo "✅ GUI has been disabled on boot"
        return 0
    else
        echo "❌ Failed to disable GUI"
        return 1
    fi
}

# Configure GUI state
configure_gui() {
    local current_state=$(get_gui_state)
    echo "Current GUI state: $current_state"
    
    # If GUI_ENABLED is set to "ask" and we're in interactive mode, ask the user
    if [ "$gui_enabled" = "ask" ] && [ "$interactive_mode" = "true" ]; then
        if [ "$current_state" = "enabled" ]; then
            if ask_yes_no "GUI is currently enabled. Would you like to disable it?"; then
                gui_enabled="no"
            else
                gui_enabled="yes"
            fi
        else
            if ask_yes_no "GUI is currently disabled. Would you like to enable it?"; then
                gui_enabled="yes"
            else
                gui_enabled="no"
            fi
        fi
    fi
    
    # Configure based on gui_enabled setting
    case "$gui_enabled" in
        "yes")
            if [ "$current_state" = "disabled" ]; then
                enable_gui
            else
                echo "GUI is already enabled"
            fi
            ;;
        "no")
            if [ "$current_state" = "enabled" ]; then
                disable_gui
            else
                echo "GUI is already disabled"
            fi
            ;;
        *)
            echo "Invalid GUI configuration value: $gui_enabled"
            return 1
            ;;
    esac
}

# Parse command line arguments
parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --enable)
                gui_enabled="yes"
                shift
                ;;
            --disable)
                gui_enabled="no"
                shift
                ;;
            --help)
                echo "Usage: $0 [--enable|--disable]"
                echo "Configure desktop GUI boot state"
                echo ""
                echo "Options:"
                echo "  --enable    Enable GUI on boot"
                echo "  --disable   Disable GUI on boot"
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
    check_systemd
    parse_args "$@"

    local current_state=$(get_gui_state)

    echo "=== GUI Configuration ==="
    configure_gui
    
    # Inform about reboot if state changed
    if [ "$(get_gui_state)" != "$current_state" ]; then
        echo "Changes will take effect after reboot"
        if [ "$interactive_mode" = "true" ] && ask_yes_no "Would you like to reboot now?"; then
            sudo reboot
        fi
    fi
}

# Run main function with arguments
main "$@"