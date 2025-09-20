#!/bin/bash
# Test results summary script for GitHub Actions workflows
# Usage: ./test-results-summary.sh <package_name> <stage1> <stage2> <build_status> [failure_phase] [failure_stage] [failure_component]

PACKAGE_NAME="$1"
STAGE1="$2"
STAGE2="$3"
BUILD_STATUS="$4"
FAILURE_PHASE="$5"
FAILURE_STAGE="$6"
FAILURE_COMPONENT="$7"

echo "=== $PACKAGE_NAME Build Test Results Summary ==="
if [ "$GITHUB_EVENT_NAME" = "pull_request" ]; then
    echo "PR from: $GITHUB_HEAD_REF_REPO_FULL_NAME"
    echo "Branch: $GITHUB_HEAD_REF"
    echo "Commit: $GITHUB_SHA"
else
    echo "Manual run from: $GITHUB_REPOSITORY"
    echo "Branch: $GITHUB_REF_NAME"
    echo "Commit: $GITHUB_SHA"
fi
echo ""
echo "Hardware: Jetson Orin"
echo "Package: $PACKAGE_NAME"
echo "Stage 1 (Package Listing): $STAGE1"
echo "Stage 2 (Package Build): $STAGE2"
echo "Overall Status: $BUILD_STATUS"

if [ "$BUILD_STATUS" = "success" ]; then
    echo "🎉 RESULT: $PACKAGE_NAME package SUCCESS on Jetson Orin"
    exit 0
else
    echo "💥 RESULT: $PACKAGE_NAME package FAILED on Jetson Orin"
    echo "Failure Phase: $FAILURE_PHASE"
    echo "Failure Stage: $FAILURE_STAGE"
    echo "Failure Component: $FAILURE_COMPONENT"
    echo ""
    echo "❌ $PACKAGE_NAME Package Test (Orin) - FAILED in STAGE: $FAILURE_PHASE > $FAILURE_STAGE > $FAILURE_COMPONENT"
    echo "Check build logs above for detailed failure information"
    exit 1
fi
