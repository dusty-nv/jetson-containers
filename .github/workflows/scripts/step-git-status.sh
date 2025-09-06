#!/bin/bash
# Git status script for GitHub Actions workflows
# Shows git repository state and recent history

echo "=== Git Status ==="
git status
echo ""
echo "=== Recent Git History (last 10 commits) ==="
git log --oneline -10
