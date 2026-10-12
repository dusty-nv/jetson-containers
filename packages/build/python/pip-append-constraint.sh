#!/usr/bin/env bash
# pip utils: append installed version of a package to the global constraints file

set -euo pipefail

pkg="${1:-}"

if [ -z "$pkg" ]; then
  echo "Usage: $0 <package-name>" >&2
  exit 1
fi

# Ensure PIP_CONSTRAINT is set
if [ -z "${PIP_CONSTRAINT:-}" ]; then
  echo "Error: PIP_CONSTRAINT environment variable not set." >&2
  exit 2
fi

# Ensure constraints file exists (create dir if needed)
mkdir -p "$(dirname "$PIP_CONSTRAINT")"
touch "$PIP_CONSTRAINT"

# Get installed version (quietly, only the version string)
ver="$(pip show "$pkg" 2>/dev/null | awk '/^Version:/{print $2}')"

if [ -z "$ver" ]; then
  echo "Error: Package '$pkg' not installed (pip show returned no version)." >&2
  exit 3
fi

line="${pkg}==${ver}"

# Check if already present
if grep -q -F "$line" "$PIP_CONSTRAINT"; then
  echo "Already present in $PIP_CONSTRAINT: $line"
else
  echo "$line" >> "$PIP_CONSTRAINT"
  echo "Appended to $PIP_CONSTRAINT: $line"
fi

echo "Current global PIP constraints:"
cat "${PIP_CONSTRAINT}"
