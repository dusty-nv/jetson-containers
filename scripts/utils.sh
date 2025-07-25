#!/bin/bash

# Enable error handling
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(dirname "$(dirname "$(readlink -fm "$0")")")"

# Global flag for quiet mode
QUIET=false

# Unified logging function
log() {
    if [ "$QUIET" = false ]; then
        echo "$@"
    fi
}

# Load environment variables from .env
load_env() {
    local env_path="${JETSON_SETUP_ENV_FILE:-$SCRIPT_DIR/.env}"

    if [ -f "${env_path}" ]; then
        set -a
        source "${env_path}"
        set +a
    else
        echo "Environment file ${env_path} not found."
        exit 1
    fi
}

file_exists() {
    local file=${1:-}
    [[ -n "$file" && -f "$file" ]]
}

check_swap_exists() {
    local swap_file="${1:-}"

    # non‑empty and regular file
    file_exists "${swap_file}" || return 1

    # check if it's active
    if swapon --noheadings --raw --show=NAME | grep -Fqx "${swap_file}"; then
        return 0
    else
        return 1
    fi
}

# Function to prompt yes/no questions
ask_yes_no() {
    while true; do
        read -p "${1} (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or y or no or n.";;
        esac
    done
}

# Pretty print functions
print_section() {
    local message=$1
    echo -e "\n\033[48;5;130m\033[97m >>> $message \033[0m"
}

pretty_print() {
    if [ "$QUIET" = "true" ]; then
        return 0
    fi

    # 1) Extract level, default to LOG
    local level="${1:-LOG}"
    shift

    # 2) Reassemble the rest of args into the message
    local message="${*}"

    # 3) Choose color based on level
    local color reset
    case "$level" in
        INFO)
            color='\033[36m'    ;;      # cyan
        WARN)
            color='\033[38;5;205m' ;;   # magenta
        ERROR)
            color='\033[1;31m'   ;;     # bold red
        LOG|*)
            color=''             ;;
    esac
    reset='\033[0m'

    # 4) Print. ERROR to stderr, others to stdout.
    if [ "$level" = "ERROR" ]; then
        printf '%b[%s]%b %s\n' "$color" "$level" "$reset" "$message" >&2
    else
        printf '%b[%s]%b %s\n' "$color" "$level" "$reset" "$message"
    fi
}

is_command_available() {
    local command=$1

    if command -v ${command} &> /dev/null; then
        pretty_print INFO "✅ ${command} is already installed."
        return 0
    else
        pretty_print WARN "❌ ${command} is NOT installed."
        return 1
    fi
}

is_docker_installed() {
    if is_command_available docker; then
        return 0
    else
        return 1
    fi
}

# Function to check Docker runtime configuration
check_docker_runtime() {
    local daemon_config="/etc/docker/daemon.json"

    if ! is_docker_installed; then
        return 1
    fi
    
    if ! file_exists $daemon_config; then
        echo "Docker ${daemon_config} file does not exist."
        return 1
    fi
    
    if grep -q '"default-runtime": "nvidia"' $daemon_config 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

l4t_root_device() {
    local root_device=$(findmnt -n -o SOURCE /)
    
    if [[ $root_device == *"nvme"* ]]; then
        echo "nvme"
    elif [[ $root_device == *"mmcblk"* ]]; then
        # eMMC devices have mmcblk0boot0 and mmcblk0boot1 partitions
        if [[ -b /dev/mmcblk0boot0 ]] && [[ -b /dev/mmcblk0boot1 ]]; then
            echo "emmc"
        else
            echo "sdcard"
        fi
    elif [[ $root_device == *"/dev/sd"* ]]; then
        echo "usb_sata"
    else
        echo "unknown"
    fi
}

is_l4t_installed_on_nvme() { [[ "$(l4t_root_device)" == "nvme" ]]; }
is_l4t_installed_on_emmc() { [[ "$(l4t_root_device)" == "emmc" ]]; }
is_l4t_installed_on_sdcard() { [[ "$(l4t_root_device)" == "sdcard" ]]; }
is_l4t_installed_on_usb() { [[ "$(l4t_root_device)" == "usb_sata" ]]; }

user_in_group() {
    id -nG "$USER" | grep -qw -- "$1"
}

# Check if script is run with sudo
check_permissions() {
    if [ "$EUID" -ne 0 ]; then 
        pretty_print ERROR "Please run as root (with sudo)"
        exit 1
    fi
}

is_true() {
    [[ "${1,,}" =~ ^(true|1|yes)$ ]]
}

ask_should_run() {
    local flag=${1:-ask}
    local question=${2:-"Would you like to run this?"}
    local interactive_mode="${INTERACTIVE_MODE:-true}"

    if is_true $flag || ([ "${flag}" = "ask" ] && is_true $interactive_mode && ask_yes_no "${question}"); then
        return 0
    else
        return 1
    fi
}

# Convert size string to bytes (e.g., "4G" to bytes)
convert_to_bytes() {
    local size=$1
    local value=${size%[GM]}
    local unit=${size#$value}
    
    case $unit in
        G)
            echo $((value * 1024 * 1024 * 1024))
            ;;
        M)
            echo $((value * 1024 * 1024))
            ;;
        *)
            echo "Error: Invalid size unit. Use M for MB or G for GB"
            exit 1
            ;;
    esac
}

ask_for_reboot() {
    local interactive_mode="${INTERACTIVE_MODE:-true}"

    if [ "$interactive_mode" = "true" ] && ask_yes_no "Would you like to reboot now?"; then
        log "Rebooting, please wait..."
        sudo reboot
    fi
}