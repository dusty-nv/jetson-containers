#!/bin/bash

#set -x # Enable debug mode

print_section() {
    local message=$1
    echo -e "\n\033[48;5;130m\033[97m >>> $message \033[0m"
}

log() {
    local level=$1
    local message=$2
    case "$level" in
        INFO)  echo -e "\033[36m[INFO]\033[0m $message" ;;      # Cyan text
        WARN)  echo -e "\033[38;5;205m[WARN]\033[0m $message" ;; # Magenta text
        ERROR) echo -e "\033[1;31m[ERROR]\033[0m $message" ;;   # Red text (bold)
        *)     echo "[LOG]  $message" ;;                        # Default log
    esac
}

print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --enable-gui          Enable GUI mode (graphical.target)"
    echo "  --disable-gui         Disable GUI mode (multi-user.target)"
    echo "  --enable-zram         Enable ZRAM swap"
    echo "  --disable-zram        Disable ZRAM swap"
    echo "  --enable-swap         Enable swap file"
    echo "  --disable-swap        Disable swap file"
    echo "  --set-max-power       Set system to MAX-N power mode"
    echo "  --test=<function>     Run a specific function for testing"
    echo "  --help                Show this help message and exit"
    echo
    echo "Example:"
    echo "  $0 --disable-gui --disable-zram --enable-swap --set-max-power"
    echo
}

parse_args() {
    TEST_FUNCTION=""

    for arg in "$@"; do
        case "$arg" in
            --enable-gui)
                ENABLE_GUI="true"
                ;;
            --disable-gui)
                DISABLE_GUI="true"
                ;;
            --enable-zram)
                ENABLE_ZRAM="true"
                ;;
            --disable-zram)
                DISABLE_ZRAM="true"
                ;;
            --enable-swap)
                ENABLE_SWAP="true"
                ;;
            --disable-swap)
                DISABLE_SWAP="true"
                ;;
            --set-max-power)
                SET_MAX_POWER="true"
                ;;
            --test=*)
                TEST_FUNCTION="${arg#*=}"
                ;;
            --help)
                print_help
                exit 0
                ;;
            *)
                log ERROR "Unknown parameter: $arg"
                exit 1
                ;;
        esac
    done
}

# Function to get available RAM in MB
get_free_ram() {
    free -m | awk '/^Mem:/ {print $7}'
}

# Function to format numbers with commas
format_number() {
    printf "%'d\n" "$1"
}


# Detect the display manager (GDM, LightDM, etc.)
detect_display_manager() {
    if systemctl list-units --type=service | grep -q 'gdm.service'; then
        echo "gdm"
    elif systemctl list-units --type=service | grep -q 'lightdm.service'; then
        echo "lightdm"
    else
        echo ""
    fi
}

toggle_gui_mode() {
    print_section "1. GUI mode setting"
    # Check current GUI status
    current_target=$(systemctl get-default)

    if [[ "$current_target" == "multi-user.target" ]]; then
        gui_status="DISABLED"
        suggested_action="Enable"
        new_target="graphical.target"
    else
        gui_status="ENABLED"
        suggested_action="Disable"
        new_target="multi-user.target"
    fi

    # Handle --enable/disable-gui argument
    if [[ "$1" == "--enable-gui" ]]; then
        if [[ "$gui_status" == "ENABLED" ]]; then
            echo "✅ GUI is already enabled. No changes needed."
            return 0
        fi
        new_target="graphical.target"
        suggested_action="Enable"
    elif [[ "$1" == "--disable-gui" ]]; then
        if [[ "$gui_status" == "DISABLED" ]]; then
            echo "✅ GUI is already disabled. No changes needed."
            return 0
        fi
        new_target="multi-user.target"
        suggested_action="Disable"
    fi

    # Display current status
    log INFO "🔎 Current GUI Status: $gui_status"
    log INFO "   Suggested Action: $suggested_action GUI mode"
    log INFO "   (multi-user.target = No GUI, graphical.target = GUI enabled)"
    echo

    # Ask user to switch mode
    read -p "Would you like to $suggested_action the GUI? (y/N): " confirm
    confirm=${confirm,,}  # Convert to lowercase

    if [[ "$confirm" != "y" && "$confirm" != "yes" ]]; then
        echo "❌ Operation aborted by user."
        return 1
    fi

    # Measure RAM before switching
    ram_before=$(get_free_ram)

    # Apply the GUI mode change
    log INFO "🔄 Switching GUI mode..."
    sudo systemctl set-default "$new_target"

    if [[ "$new_target" == "multi-user.target" ]]; then
        # If disabling GUI, stop the display manager immediately
        display_manager=$(detect_display_manager)

        if [[ -n "$display_manager" ]]; then
            echo "⏳ Stopping $display_manager to free up RAM immediately..."
            sudo systemctl stop "$display_manager"
        fi

        # Kill any remaining GUI sessions
        echo "🔪 Killing remaining GUI processes..."
        sudo pkill -f Xorg
        sudo pkill -f gnome-session
        sudo pkill -f plasmashell
        sudo pkill -f xfwm4
        sudo pkill -f lxsession
    fi

    # Measure RAM after switching
    sleep 5  # Give the system time to release memory
    ram_after=$(get_free_ram)

    # Calculate RAM difference
    ram_diff=$((ram_after - ram_before))

    # Format numbers with commas
    formatted_ram_before=$(format_number "$ram_before")
    formatted_ram_after=$(format_number "$ram_after")
    formatted_ram_diff=$(format_number "${ram_diff#-}")  # Remove negative sign

    # Display results
    log INFO "✅ GUI mode switched to: $new_target"
    log INFO "💾 Available RAM before: ${formatted_ram_before}MB"
    log INFO "💾 Available RAM after:  ${formatted_ram_after}MB"

    if [[ "$ram_diff" -gt 0 ]]; then
        echo "🚀 Freed RAM:  ${formatted_ram_diff}MB (Disabling GUI)"
    elif [[ "$ram_diff" -lt 0 ]]; then
        echo "⬇️  Additional RAM used: ${formatted_ram_diff}MB (Enabling GUI)"
    else
        echo "ℹ️  No significant RAM change detected."
    fi

    log INFO "🔄 To apply changes, reboot your system: sudo reboot"
}


toggle_zram() {
    print_section "2. ZRAM setting"

    if systemctl is-enabled nvzramconfig &> /dev/null; then
        log INFO "ZRAM (nvzramconfig) is enabled."
        zram_status="ENABLED"
        action="disable"
    else
        log INFO "zram (nvzramconfig) is disabled🔕."
        zram_status="DISABLED"
        action="enable"
    fi

    # Handle --enable/disable-zram argument
    if [[ "$1" == "--enable-zram" ]]; then
        if [[ "$zram_status" == "ENABLED" ]]; then
            echo "✅ ZRAM is already enabled. No changes needed."
            return 0
        fi
        action="enable"
    elif [[ "$1" == "--disable-zram" ]]; then
        if [[ "$zram_status" == "DISABLED" ]]; then
            echo "✅ ZRAM is already disabled🔕. No changes needed."
            return 0
        fi
        action="disable"
    fi

    echo "🔄 Current zram status: $zram_status"
    read -p "Would you like to $action zram? (y/N): " confirm
    confirm=${confirm,,}  # Convert to lowercase

    if [[ "$confirm" != "y" && "$confirm" != "yes" ]]; then
        echo "❌ Operation aborted."
        return 1
    fi

    # Toggle zram
    if [[ "$zram_status" == "ENABLED" ]]; then
        echo "Disabling zram..."
        if sudo systemctl disable nvzramconfig && sudo systemctl stop nvzramconfig; then
            echo "✅ zram disabled🔕 successfully."
        else
            echo "❌ Failed to disable zram."
            return 1
        fi
        sudo swapoff -a
    else
        echo "Enabling zram..."
        if sudo systemctl enable nvzramconfig && sudo systemctl start nvzramconfig; then
            echo "✅ zram enabled successfully."
        else
            echo "❌ Failed to enable zram."
            return 1
        fi
    fi
    sleep 3
    free -h
    return 0
}

toggle_swap() {
    print_section "3. Swap-file setting"
    log WARN "A big issue that destoys /etc/fstab found. Disabling this sectin for now."

    # # Step 1: Get a list of NVMe-backed mount points
    # nvme_mounts=($(lsblk -nr -o NAME,MOUNTPOINT | awk '/^nvme/ && $2 != "" {print $2}'))
    # log INFO "nvme_mounts: $nvme_mounts"

    # # Step 2: Find all active swap files
    # all_swap=$(swapon --show --noheadings | awk '$1 !~ /\/dev\/zram/ {print $1}')
    # log INFO "all_swap: $all_swap"

    # # Step 3: Exclude swap files that are inside an NVMe-backed filesystem
    # non_zram_non_nvme_swap=()

    # for swap in $all_swap; do
    #     # Check if the swap file is inside an NVMe mount point
    #     is_nvme_backed=false
    #     for mount in "${nvme_mounts[@]}"; do
    #         if [[ "$swap" == "$mount"* ]]; then
    #             is_nvme_backed=true
    #             log INFO "$swap is NVMe backed"
    #             break
    #         fi
    #     done

    #     # If it's NOT NVMe-backed, add to the filtered list
    #     if [[ "$is_nvme_backed" == false ]]; then
    #         non_zram_non_nvme_swap+=("$swap")
    #     fi
    # done

    # # Step 4: Print the filtered swap files
    # if [[ ${#non_zram_non_nvme_swap[@]} -eq 0 ]]; then
    #     echo "✅ No non-ZRAM, non-NVMe swap files found."
    # else
    #     echo "🐌 Non-ZRAM, non-NVMe swap files detected:"
    #     printf "%s\n" "${non_zram_non_nvme_swap[@]}"

    #     # Step 5. Disable and remove each non-ZRAM swap file
    #     echo "⚠️  Disabling and deleting non-ZRAM, non-NVMe swap files🐌 ..."
    #     for swap in $non_zram_non_nvme_swap; do
    #         echo "🚫 Disabling swap: $swap"
    #         sudo swapoff "$swap"

    #         if [[ -f "$swap" ]]; then
    #             echo "🗑️  Deleting swap file: $swap"
    #             sudo rm -f "$swap"
    #         else
    #             echo "⚠️  $swap is not a file, skipping deletion."
    #         fi
    #     done

    #     # Verify swap is fully disabled
    #     echo "🔍 Checking remaining swap..."
    #     swapon --show --noheadings

    #     if swapon --show --noheadings | grep -qv '/dev/zram'; then
    #         log ERROR "❌ Some swap files are still active!"
    #         return 1
    #     else
    #         log INFO "✅ All non-ZRAM swap files removed successfully!"
    #     fi
    # fi

    # if swapon --show --noheadings | grep -qv '/dev/zram'; then
    #     echo "There is a non-zram swap device active🏃."
    #     swapon --show | grep -v '/dev/zram'
    #     swap_status="ENABLED"
    #     action="disable"
    # else
    #     echo "No swap file present (or only zram swap is active)🔕."
    #     swap_status="DISABLED"
    #     action="enable"
    # fi

    # # Handle --enable/disable-swap argument
    # if [[ "$1" == "--enable-swap" ]]; then
    #     if [[ "$swap_status" == "ENABLED" ]]; then
    #         echo "✅ Swap file is already enabled. No changes needed."
    #         return 0
    #     fi
    #     action="enable"
    # elif [[ "$1" == "--disable-swap" ]]; then
    #     if [[ "$swap_status" == "DISABLED" ]]; then
    #         echo "✅ Swap file is already disabled🔕. No changes needed."
    #         return 0
    #     fi
    #     action="disable"
    # fi

    # echo "🔄 Current swap status: $swap_status"
    # read -p "Would you like to $action swap file? (y/N): " confirm
    # confirm=${confirm,,}  # Convert to lowercase

    # if [[ "$confirm" != "y" && "$confirm" != "yes" ]]; then
    #     echo "❌ Operation aborted."
    #     return 1
    # fi

    # # Determine action and execute it
    # if [[ "$action" == "enable" ]]; then
    #     echo "🚀 Enabling swap files..."

    #     # Detect NVMe mount points and set default swap file location
    #     nvme_mount_points=($(lsblk -nr -o NAME,MOUNTPOINT | awk '/^nvme/ && $2 != "" {print $2}'))

    #     if [[ ${#nvme_mount_points[@]} -gt 0 ]]; then
    #         default_swap_location="${nvme_mount_points[0]}/swapfile"
    #     else
    #         default_swap_location="/var/swapfile"  # Fallback if no NVMe is found
    #     fi

    #     # Ask the user for the swap file location (default to NVMe)
    #     echo -n "Enter swap file location [Default: $default_swap_location]: "
    #     read -e -i "$default_swap_location" swap_file

    #     # Get total system memory in GB
    #     total_mem_gb=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)

    #     # Calculate default swap size (half of total memory, rounded)
    #     default_swap_size=$(( (total_mem_gb + 2) / 2 ))  # Ensure rounding up for odd numbers

    #     # Ask the user for swap size (default: 4GB)
    #     echo -n "Enter swap file size in GB [Default: $default_swap_size]: "
    #     read -e -i "$default_swap_size" swap_size

    #     # Confirm user input
    #     echo "🔄 Creating a swap file at: $swap_file"
    #     echo "🔄 Swap file size: ${swap_size}GB"
    #     read -p "Proceed? (y/N): " confirm
    #     confirm=${confirm,,}  # Convert to lowercase

    #     if [[ "$confirm" != "y" && "$confirm" != "yes" ]]; then
    #         echo "❌ Operation aborted."
    #         exit 1
    #     fi

    #     # Create the swap file
    #     echo "🛠️ Allocating swap file..."
    #     sudo fallocate -l "${swap_size}G" "$swap_file" || sudo dd if=/dev/zero of="$swap_file" bs=1G count="$swap_size" status=progress
    #     sudo chmod 600 "$swap_file"
    #     sudo mkswap "$swap_file"
    #     sudo swapon "$swap_file"

    #     # Verify swap activation
    #     echo "🔍 Checking active swap files..."
    #     swapon --show

    #     # Add to /etc/fstab if not already present
    #     if ! grep -q "$swap_file" /etc/fstab; then

    #         # Get UUID of the swap file (if it exists)
    #         swap_uuid=$(blkid -s UUID -o value "$swap_file")

    #         # If blkid fails, use the swap file path instead
    #         if [[ -z "$swap_uuid" ]]; then
    #             echo "⚠️  No UUID found for $swap_file, using direct path instead."
    #             swap_entry="$swap_file none swap sw 0 0"
    #         else
    #             swap_entry="UUID=$swap_uuid none swap sw 0 0"
    #         fi

    #         echo "$swap_entry" | sudo tee -a /etc/fstab
    #         echo "✅ Swap file added to /etc/fstab for persistence."
    #     else
    #         echo "⚠️ Swap file already exists in /etc/fstab."
    #     fi

    #     echo "✅ Swap setup complete!"

    # elif [[ "$action" == "disable" ]]; then
    #     echo "🚫 Disabling swap files..."

    #     # Iterate over all non-ZRAM, non-NVMe swap files
    #     for swap in $(swapon --show --noheadings | awk '$1 !~ /\/dev\/zram/ && $1 !~ /\/dev\/nvme/ {print $1}'); do
    #         echo "🔄 Disabling swap: $swap"
    #         sudo swapoff "$swap" && echo "✅ Swap disabled: $swap" || echo "❌ Failed to disable swap: $swap"
    #     done
    # fi

    # # Verify swap status after the operation
    # echo "🔍 Checking updated swap status..."
    # swapon --show --noheadings

    # if swapon --show --noheadings | grep -q .; then
    #     echo "Swap is active.🏃"
    # else
    #     echo "All swap files have been disabled.🔕"
    # fi

    # sleep 3
    # free -h
    return 0
}

set_max_power() {
    print_section "4. NV power mode setting"
    log INFO "🔋 Setting Jetson to MAXN Power Mode..."

    # Get the current power mode
    current_mode=$(sudo nvpmodel -q | awk -F': ' '/NV Power Mode/ {print $2}' | tr -d ' ')
    log INFO "current_mode: $current_mode"

    if [[ "$current_mode" =~ ^MAXN  ]]; then
        log WARN "✅ Jetson is already in $current_mode mode. Skipping nvpmodel change."
    else
        # Confirm before proceeding
        read -p "⚠️  This will set your Jetson to maximum power mode (MAXN). Proceed? (y/N): " confirm
        confirm=${confirm,,}  # Convert input to lowercase

        if [[ "$confirm" != "y" && "$confirm" != "yes" ]]; then
            log WARN "❌ Operation aborted."
            return 1
        fi

        local maxn_id
        maxn_id=$(awk -F'[= >]+' '/< POWER_MODEL/ {id=$4} /NAME=MAXN/ {print id; exit}' /etc/nvpmodel.conf)

        if [[ -n "$maxn_id" ]]; then
            log INFO "✅ MAXN Power Model ID: $maxn_id"
        else
            log WARN "❌ No MAXN mode found in /etc/nvpmodel.conf" >&2
            return 1
        fi

        log INFO "🛠️ Applying MAXN Power Mode..."
        log WARN "⚠️ Rebooting may be required and you may get asked to type YES to reboot."
        sudo nvpmodel -m "$maxn_id"
        sleep 1  # Allow time for mode change to take effect

        # Check if MAXN mode is now active
        new_mode=$(sudo nvpmodel -q | awk -F': ' '/NV Power Mode/ {print $2}' | tr -d ' ')

        if [[ "$new_mode" == MAXN* ]]; then
            log INFO "✅ Jetson is now running in MAXN mode ($new_mode)."
            return 0
        fi
    fi

    # Boost clocks
    log INFO "🚀 Applying Jetson Clocks for max performance..."
    sudo jetson_clocks

    log INFO "✅ Jetson is now running in MAXN mode with boosted clocks!"
}

main() {

    parse_args "$@"

    if [[ -n "$TEST_FUNCTION" ]]; then
        # Check if the function exists before calling
        if declare -F "$TEST_FUNCTION" > /dev/null; then
            log WARN "Running test for function: $TEST_FUNCTION"
            "$TEST_FUNCTION"  # Call the function dynamically
            exit 0
        else
            log ERROR "Function '$TEST_FUNCTION' not found."
            exit 1
        fi
    fi

    # Step 1. Toggle GUI mode (graphical <-> multi-user)
    if [[ "$ENABLE_GUI" == "true" ]]; then
        toggle_gui_mode --enable-gui
    elif [[ "$DISABLE_GUI" == "true" ]]; then
        toggle_gui_mode --disable-gui
    else
        toggle_gui_mode
    fi

    # Step 2. Toggle ZRAM mode (-> off)
    if [[ "$ENABLE_ZRAM" == "true" ]]; then
        toggle_zram --enable-zram
    elif [[ "$DISABLE_ZRAM" == "true" ]]; then
        toggle_zram --disable-zram
    else
        toggle_zram
    fi

    # Step 3. Toggle Swap File (-> ON)
    if [[ "$ENABLE_SWAP" == "true" ]]; then
        toggle_swap --enable-swap
    elif [[ "$DISABLE_SWAP" == "true" ]]; then
        toggle_swap --disable-swap
    else
        toggle_swap
    fi

    # Step 4. SET_MAX_POWER (-> ON)
    if [[ "$SET_MAX_POWER" == "true" ]]; then
        set_max_power
    fi
}

# Execute main function
main "$@"