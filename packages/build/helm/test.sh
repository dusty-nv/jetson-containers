#!/usr/bin/env bash
set -euo pipefail   # -u catches unset vars, -o pipefail handles pipes properly

echo "🔍  Testing Helm…"
helm version --short          # prints vX.Y.Z+gHASH
echo "✅  Helm OK"
