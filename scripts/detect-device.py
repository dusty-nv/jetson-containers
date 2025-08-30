#!/usr/bin/env bash
set -euo pipefail

# Get first and last 'compatible' strings (property is NUL-separated)
mapfile -t COMP < <(tr '\0' '\n' </proc/device-tree/compatible | sed '/^$/d')
CARRIER="${COMP[0]}"
SOC="${COMP[${#COMP[@]}-1]}"

# Memory from /proc/meminfo (kB) → MB + GB
mem_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
mem_mb=$(( mem_kb / 1024 ))
mem_gb=$(awk -v kb="$mem_kb" 'BEGIN{printf "%.1f", kb/1048576}')

echo "Carrier: $CARRIER"
echo "SoC:     $SOC"
echo "Memory:  ${mem_mb} MB (${mem_gb} GB)"