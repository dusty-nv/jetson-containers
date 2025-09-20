#!/usr/bin/env bash

set -euo pipefail

# Inputs:
#   BASE_REF  - base branch name (e.g., 'dev') for PR diff (optional)
# Outputs:
#   Writes a JSON array of package names to $GITHUB_OUTPUT as 'packages', if available.
#   Also echoes the JSON to stdout for debugging.

BASE_REF_ENV=${BASE_REF:-}

if [[ -z "$BASE_REF_ENV" ]]; then
  json='[]'
  echo "$json"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "packages=$json" >> "$GITHUB_OUTPUT"
  fi
  exit 0
fi

git fetch --no-tags --prune origin "$BASE_REF_ENV"

mapfile -t candidates < <(
  git diff --name-only "origin/${BASE_REF_ENV}"...HEAD \
    | awk -F/ '$1=="packages" { if (NF>=3) print $3; else if (NF==2) print $2 }' \
    | sort -u
)

packages=()
for p in "${candidates[@]}"; do
  [[ -z "$p" || "$p" == *.* ]] && continue
  if [[ -d "packages/$p" ]] || compgen -G "packages/*/$p" >/dev/null; then
    packages+=("$p")
  fi
done

if (( ${#packages[@]} == 0 )); then
  json='[]'
  echo "$json"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "packages=$json" >> "$GITHUB_OUTPUT"
  fi
  exit 0
fi

json='['
sep=''
for p in "${packages[@]}"; do
  json+="$sep\"$p\""
  sep=','
done
json+=']'

echo "$json"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "packages=$json" >> "$GITHUB_OUTPUT"
fi


