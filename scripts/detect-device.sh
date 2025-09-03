#!/usr/bin/env bash
set -euo pipefail

# Get first and last 'compatible' strings (property is NULL-separated)
mapfile -t COMP < <(tr '\0' '\n' </proc/device-tree/compatible | sed '/^$/d')
if [ "${#COMP[@]}" -eq 0 ]; then
    echo "Error: /proc/device-tree/compatible is empty or missing." >&2
    exit 1
fi
CARRIER="${COMP[0]}"
SOC="${COMP[${#COMP[@]}-1]}"

# Memory from /proc/meminfo (kB) → MB + GB
mem_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
if [[ -z "$mem_kb" || ! "$mem_kb" =~ ^[0-9]+$ ]]; then
    mem_mb="unknown"
    mem_gb="unknown"
else
    mem_mb=$(( mem_kb / 1024 ))
    mem_gb=$(awk -v kb="$mem_kb" 'BEGIN{printf "%.1f", kb/1048576}')
fi

echo "Carrier: $CARRIER"
echo "SoC:     $SOC"
echo "Memory:  ${mem_mb} MB (${mem_gb} GB)"
