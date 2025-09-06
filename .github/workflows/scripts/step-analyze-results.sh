#!/bin/bash
# Analyze build results script for GitHub Actions workflows
# Usage: ./analyze-results.sh <package_name> <stage1> <stage2> <build_status> [failure_phase] [failure_stage] [failure_component]

PACKAGE_NAME="$1"
STAGE1="$2"
STAGE2="$3"
BUILD_STATUS="$4"
FAILURE_PHASE="$5"
FAILURE_STAGE="$6"
FAILURE_COMPONENT="$7"

echo "=== Build Analysis ==="
echo "Stage 1 (Package Listing): $STAGE1"
echo "Stage 2 (Package Build): $STAGE2"
echo "Overall Build Status: $BUILD_STATUS"

if [ "$BUILD_STATUS" = "success" ]; then
    echo "🎉 SUCCESS: $PACKAGE_NAME package built successfully on Jetson Orin"
else
    echo "💥 FAILURE: $PACKAGE_NAME package build failed on Jetson Orin"
    echo "Check the build log above for detailed error information"
fi
