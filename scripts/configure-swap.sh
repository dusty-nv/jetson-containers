#!/bin/bash

# Enable error handling
set -euo pipefail

. scripts/utils.sh

# Load environment variables from .env
load_env

# Default values
interactive_mode="${INTERACTIVE_MODE:-true}"
# Use SWAP_OPTIONS_SIZE from the .env file if present, otherwise default to 8G
swap_size_num="${SWAP_OPTIONS_SIZE:-8}"
swap_size="${swap_size_num}G"
# Use SWAP_OPTIONS_DISABLE_ZRAM to determine if zRAM should be disabled by default
zram_disabled="${SWAP_OPTIONS_DISABLE_ZRAM:-false}"
zram_enabled="ask"
zram_size="${ZRAM_SIZE:-4G}"
# Use SWAP_OPTIONS_PATH if specified, otherwise default to /swapfile
swap_file_path="${SWAP_OPTIONS_PATH:-/swapfile}"

if is_true $zram_disabled; then
    zram_enabled="no"
fi

# On the SD Card we prefer to use the zram only
if ! is_true $zram_enabled && is_l4t_installed_on_sdcard; then
    zram_enabled="yes"
fi

# Check if swap file exists and is active
check_swap_file_status() {
    if file_exists "$swap_file_path"; then
        if check_swap_exists "$swap_file_path"; then
            echo "active"
        else
            echo "inactive"
        fi
    else
        echo "missing"
    fi
}

# Create and enable swap file
# Helper function for robust swap file cleanup
cleanup_swap_file() {
    local file="$1"
    [[ -z "${file}" ]] && return 0
    
    sync 2>/dev/null || true
    sudo swapoff "${file}" 2>/dev/null || true
    sudo rm -f "${file}" 2>/dev/null || true
}

# Create and enable swap file with safety checks
setup_swap_file() {
    local swap_state=$(check_swap_file_status)
    local size_bytes=$(convert_to_bytes "${swap_size}")
    
    # Validate inputs
    if [[ -z "${swap_file_path}" ]]; then
        echo "❌ Error: swap_file_path is not set"
        return 1
    fi
    
    if [[ -z "${size_bytes}" ]] || [[ "${size_bytes}" -le 0 ]]; then
        echo "❌ Error: Invalid swap size: ${swap_size}"
        return 1
    fi
    
    # Modern bash (4.0+) supports 64-bit arithmetic, but let's be safe for very large values
    # Only restrict truly excessive sizes (>1TB) that might indicate input errors
    local max_reasonable_swap=$((1024 * 1024 * 1024 * 1024))  # 1TB limit
    if [[ "${size_bytes}" -gt "${max_reasonable_swap}" ]]; then
        echo "❌ Error: Swap size unreasonably large (>1TB): ${swap_size}"
        echo "   This might indicate an input parsing error"
        return 1
    fi
    
    # Validate swap file path (should be absolute and in safe location)
    if [[ "${swap_file_path}" != /* ]]; then
        echo "❌ Error: swap_file_path must be absolute: ${swap_file_path}"
        return 1
    fi
    
    # Ensure path is in a safe location (not root, boot, etc.)
    case "${swap_file_path}" in
        /boot/*|/proc/*|/sys/*|/dev/*|/run/*)
            echo "❌ Error: Unsafe swap file location: ${swap_file_path}"
            return 1
            ;;
    esac
    
    # Get directory and check if it exists
    local swap_dir=$(dirname "${swap_file_path}")
    if [[ ! -d "${swap_dir}" ]]; then
        echo "❌ Error: Directory does not exist: ${swap_dir}"
        return 1
    fi
    
    # Check write permissions on directory
    if [[ ! -w "${swap_dir}" ]] && ! sudo test -w "${swap_dir}"; then
        echo "❌ Error: No write permission for directory: ${swap_dir}"
        return 1
    fi
    
    # Check filesystem size limits and available space
    local filesystem_size=$(df --output=size -B1 "${swap_dir}" | tail -n1)
    local max_reasonable_size=$((filesystem_size / 2))  # Don't use more than 50% of filesystem
    
    if [[ "${size_bytes}" -gt "${max_reasonable_size}" ]]; then
        echo "❌ Error: Swap size too large (max recommended: $(numfmt --to=iec ${max_reasonable_size}))"
        echo "   Requested: $(numfmt --to=iec ${size_bytes})"
        echo "   Filesystem: $(numfmt --to=iec ${filesystem_size})"
        return 1
    fi
    
    # Check available space (add 10% buffer)
    # For large values, use external calculation to be safe
    local required_space
    if command -v bc >/dev/null 2>&1; then
        required_space=$(echo "${size_bytes} * 1.1" | bc | cut -d. -f1)
    else
        # Fallback: use bash arithmetic with overflow protection
        if [[ "${size_bytes}" -gt 1000000000000 ]]; then  # >1TB
            # For very large sizes, approximate without overflow risk
            required_space="${size_bytes}"  # Skip buffer for huge sizes
        else
            required_space=$((size_bytes + size_bytes/10))
        fi
    fi
    
    local available_space=$(df --output=avail -B1 "${swap_dir}" | tail -n1)
    
    # Compare using string comparison for large numbers
    if [[ "${required_space}" -gt "${available_space}" ]] 2>/dev/null || 
       (command -v bc >/dev/null 2>&1 && (( $(echo "${required_space} > ${available_space}" | bc -l) ))); then
        echo "❌ Error: Insufficient disk space"
        echo "   Required: $(numfmt --to=iec ${required_space} 2>/dev/null || echo "${required_space} bytes")"
        echo "   Available: $(numfmt --to=iec ${available_space} 2>/dev/null || echo "${available_space} bytes")"
        return 1
    fi
    
    # If swap file exists, disable and remove it first
    if [ "${swap_state}" != "missing" ]; then
        log "Disabling existing swap file: ${swap_file_path}"
        cleanup_swap_file "${swap_file_path}"
        # Remove from fstab if present
        sudo sed -i "\\#^${swap_file_path}#d" /etc/fstab
    fi
    
    # Create swap only when not installed on sdcard
    if ! is_l4t_installed_on_sdcard; then
        log "Creating new swap file of size ${swap_size}..."
        
        # Calculate count more safely
        local count_mb=$((size_bytes/1024/1024))
        if [[ $((count_mb * 1024 * 1024)) -ne "${size_bytes}" ]]; then
            echo "❌ Error: Size calculation overflow or precision loss"
            return 1
        fi
        
        # Use fallocate if available (faster and safer than dd)
        if command -v fallocate >/dev/null 2>&1; then
            if ! sudo fallocate -l "${size_bytes}" "${swap_file_path}"; then
                echo "❌ Error: Failed to create swap file with fallocate"
                return 1
            fi
        else
            # Fallback to dd with additional safety
            if ! sudo dd if=/dev/zero of="${swap_file_path}" bs=1M count="${count_mb}" status=progress conv=fsync 2>/dev/null; then
                echo "❌ Error: Failed to create swap file with dd"
                # Clean up partial file
                cleanup_swap_file "${swap_file_path}"
                return 1
            fi
        fi
        
        # Set proper permissions
        if ! sudo chmod 600 "${swap_file_path}"; then
            echo "❌ Error: Failed to set swap file permissions"
            cleanup_swap_file "${swap_file_path}"
            return 1
        fi
        
        # Initialize swap
        if ! sudo mkswap "${swap_file_path}" >/dev/null; then
            echo "❌ Error: Failed to initialize swap file"
            cleanup_swap_file "${swap_file_path}"
            return 1
        fi
        
        # Enable swap
        if ! sudo swapon "${swap_file_path}"; then
            echo "❌ Error: Failed to enable swap file"
            cleanup_swap_file "${swap_file_path}"
            return 1
        fi
        
        # Add to fstab if not already present (using awk for precise matching)
        if ! awk -v path="${swap_file_path}" '$1 == path && $3 == "swap" {found=1} END {exit !found}' /etc/fstab 2>/dev/null; then
            if ! echo "${swap_file_path} none swap sw,pri=1 0 0" | sudo tee -a /etc/fstab >/dev/null; then
                echo "⚠️  Warning: Failed to add swap to fstab (swap is still active)"
            fi
        fi
        
        # Final verification
        if check_swap_exists "${swap_file_path}"; then
            echo "✅ Swap file setup complete ($(numfmt --to=iec ${size_bytes}))"
            return 0
        else
            echo "❌ Failed to verify swap file setup"
            return 1
        fi
    else
        log "Skipping swap file creation (L4T installed on SD card)"
        return 0
    fi
}

# Check if zRAM is enabled
check_zram_state() {
    if swapon --noheadings --raw --show=NAME | grep -q 'zram'; then
        echo "active"
    elif file_exists /etc/modules-load.d/zram.conf; then
        echo "configured"
    else
        echo "disabled"
    fi
}

# Setup zRAM
setup_zram() {
    local zram_state=$(check_zram_state)
    local size_bytes=$(convert_to_bytes "${zram_size}")
    
    # If zRAM should be enabled
    if is_true "${zram_enabled}"; then
        # Create configuration files if they don't exist
        echo "zram" | sudo tee /etc/modules-load.d/zram.conf
        echo "options zram num_devices=1" | sudo tee /etc/modprobe.d/zram.conf
        
        # Create udev rule for zRAM configuration
        cat << EOF | sudo tee /etc/udev/rules.d/99-zram.rules
KERNEL=="zram0", ATTR{disksize}="${size_bytes}", TAG+="systemd"
EOF
        
        # Create systemd service for zRAM
        cat << EOF | sudo tee /etc/systemd/system/zram.service
[Unit]
Description=zRAM setup
After=local-fs.target

[Service]
Type=oneshot
ExecStartPre=/usr/sbin/modprobe zram
ExecStartPre=/usr/bin/bash -c 'echo ${size_bytes} > /sys/block/zram0/disksize'
ExecStart=/sbin/mkswap /dev/zram0
ExecStart=/sbin/swapon -p 5 /dev/zram0
ExecStop=/sbin/swapoff /dev/zram0
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
        
        # Enable and start zRAM
        sudo modprobe zram
        sudo systemctl daemon-reload
        sudo systemctl enable zram
        sudo systemctl start zram
        echo
        
        if [ "$(check_zram_state)" = "active" ]; then
            echo "✅ zRAM setup complete"
            return 0
        else
            echo "❌ Failed to setup zRAM"
            return 1
        fi
    else
        # Disable zRAM if it's enabled
        if [ "${zram_state}" != "disabled" ]; then
            sudo systemctl stop zram
            sudo systemctl disable zram
            sudo swapoff /dev/zram0 2>/dev/null || true
            sudo rmmod zram 2>/dev/null || true
            sudo rm -f /etc/modules-load.d/zram.conf /etc/modprobe.d/zram.conf \
                      /etc/udev/rules.d/99-zram.rules /etc/systemd/system/zram.service
            echo "✅ zRAM has been disabled"
        else
            echo "zRAM is already disabled"
        fi
    fi
}

# Configure memory settings
configure_memory() {
    # Setup swap file
    if ! is_l4t_installed_on_sdcard; then
        echo "=== Configuring Swap File ==="
    fi

    setup_swap_file
    
    # Configure zRAM if needed
    echo -e "\n=== Configuring zRAM ==="
    if [ "${zram_enabled}" = "ask" ] && [ "$interactive_mode" = "true" ]; then
        if ask_yes_no "Would you like to disable zRAM?"; then
            zram_enabled="no"
        else
            zram_enabled="yes"
        fi
    fi

    setup_zram
    
    # Display current memory configuration
    echo -e "\n=== Current Memory Configuration ==="
    free -h

    # Display swap configuration if supported
    if ! is_l4t_installed_on_sdcard; then
        echo -e "\nSwap configuration:"
        swapon -s
    fi

    echo
    echo "✅ Memory setup complete"
}

# Parse command line arguments
parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --swap-size=*)
                swap_size="${1#*=}"
                shift
                ;;
            --zram-size=*)
                zram_size="${1#*=}"
                shift
                ;;
            --enable-zram)
                zram_enabled="yes"
                shift
                ;;
            --disable-zram)
                zram_enabled="no"
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Configure swap file and zRAM"
                echo ""
                echo "Options:"
                echo "  --swap-size=SIZE  Set swap file size (e.g., 8G, 4096M)"
                echo "  --zram-size=SIZE  Set zRAM size (e.g., 4G, 2048M)"
                echo "  --enable-zram     Enable zRAM"
                echo "  --disable-zram    Disable zRAM"
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

# Main function
main() {
    check_permissions
    parse_args "$@"
    configure_memory
}

# Run main function with arguments
main "$@"