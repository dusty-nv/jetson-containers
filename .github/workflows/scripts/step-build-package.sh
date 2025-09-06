#!/bin/bash
# Build package script for GitHub Actions workflows
# Usage: ./build-package.sh <package_name>
# Example: ./build-package.sh vllm

PACKAGE_NAME="$1"

if [ -z "$PACKAGE_NAME" ]; then
    echo "Error: Package name is required"
    echo "Usage: $0 <package_name>"
    exit 1
fi
echo "=== Testing $PACKAGE_NAME package on Jetson Orin ==="
chmod +x ./build.sh 
chmod +x ./run.sh

# Stage 1: List packages
echo "🔍 STAGE 1: Listing available packages..."
if ./build.sh --list-packages; then
    echo "✅ STAGE 1 PASSED: Package listing successful"
    echo "stage1=passed" >> $GITHUB_OUTPUT
else
    echo "❌ STAGE 1 FAILED: Package listing failed"
    echo "stage1=failed" >> $GITHUB_OUTPUT
    exit 1
fi

# Stage 2: Build package
echo ""
echo "🔨 STAGE 2: Building $PACKAGE_NAME package..."
echo "Command: ./build.sh --build-flags='--no-cache' --no-github-api $PACKAGE_NAME"

# Capture build output with timestamps
if timeout 7200 ./build.sh --build-flags="--no-cache" --no-github-api "$PACKAGE_NAME" 2>&1 | tee build.log; then
    # Check if the build actually succeeded by looking for failure indicators
    if grep -i "failed building\|build failed\|error.*failed" build.log; then
        echo "❌ STAGE 2 FAILED: Build script completed but build failed"
        echo "stage2=failed" >> $GITHUB_OUTPUT
        echo "build_status=failed" >> $GITHUB_OUTPUT

        # Analyze failure using log structure
        echo ""
        echo "🔍 BUILD FAILURE ANALYSIS:"
        echo "Last 50 lines of build log:"
        tail -50 build.log || echo "No build log available"

        # Initialize failure detection variables
        FAILURE_STAGE="unknown"
        FAILURE_COMPONENT="unknown"
        FAILURE_PHASE="unknown"

        # Check if logs directory exists and find the latest session
        if [ -d "logs" ]; then
            echo ""
            echo "📁 Analyzing structured log files..."

            # Find the latest log session
            LATEST_SESSION=$(ls -t logs/ | head -1)
            if [ -n "$LATEST_SESSION" ]; then
                echo "Latest log session: $LATEST_SESSION"
                cd "logs/$LATEST_SESSION"

                # Check build phase logs
                if [ -d "build" ]; then
                    echo "🔨 Analyzing build phase logs..."

                    # Find the last build log file (highest numbered)
                    LAST_BUILD_LOG=$(ls build/*.txt 2>/dev/null | sort -V | tail -1)
                    if [ -n "$LAST_BUILD_LOG" ]; then
                        echo "Last build log: $LAST_BUILD_LOG"

                        # Extract package name from log filename (format: XXoX_package_name.txt)
                        PACKAGE_NAME=$(basename "$LAST_BUILD_LOG" .txt | sed 's/^[0-9]*o[0-9]*_//')
                        echo "Failed package: $PACKAGE_NAME"

                        # Check for errors in the last build log
                        if grep -i "failed building\|build failed\|error.*failed\|exit status.*1" "$LAST_BUILD_LOG"; then
                            FAILURE_PHASE="build"
                            FAILURE_STAGE="build"
                            FAILURE_COMPONENT="$PACKAGE_NAME"
                            echo "❌ Build failure detected in package: $PACKAGE_NAME"
                        fi
                    fi
                fi

                # Check test phase logs
                if [ -d "test" ]; then
                    echo "🧪 Analyzing test phase logs..."

                    # Find the last test log file
                    LAST_TEST_LOG=$(ls test/*.txt 2>/dev/null | sort -V | tail -1)
                    if [ -n "$LAST_TEST_LOG" ]; then
                        echo "Last test log: $LAST_TEST_LOG"

                        # Extract package name from test log filename
                        TEST_PACKAGE_NAME=$(basename "$LAST_TEST_LOG" .txt | sed 's/^[0-9]*-[0-9]*_//' | sed 's/_test\.sh$//')
                        echo "Failed test package: $TEST_PACKAGE_NAME"

                        # Check for errors in the last test log
                        if grep -i "traceback\|error\|exception\|failed\|exit status.*1\|returned non-zero exit status" "$LAST_TEST_LOG"; then
                            FAILURE_PHASE="test"
                            FAILURE_STAGE="test"
                            FAILURE_COMPONENT="$TEST_PACKAGE_NAME"
                            echo "❌ Test failure detected in package: $TEST_PACKAGE_NAME"
                        fi
                    fi
                fi

                # Go back to original directory
                cd - > /dev/null
            else
                echo "No log sessions found"
            fi
        else
            echo "No logs directory found, analyzing build.log only"
        fi

        # Fallback: analyze build.log for general patterns
        if [ "$FAILURE_STAGE" = "unknown" ]; then
            echo "🔍 Fallback analysis from build.log..."

            if grep -i "timeout" build.log; then
                FAILURE_PHASE="build"
                FAILURE_STAGE="timeout"
                FAILURE_COMPONENT="build"
                echo "⏰ Timeout detected"
            elif grep -i "docker" build.log | grep -i "error\|failed"; then
                FAILURE_PHASE="build"
                FAILURE_STAGE="docker"
                FAILURE_COMPONENT="container"
                echo "❌ Docker container failure detected"
            elif grep -i "error\|failed" build.log; then
                FAILURE_PHASE="build"
                FAILURE_STAGE="build"
                FAILURE_COMPONENT="general"
                echo "❌ General build failure detected"
            else
                FAILURE_PHASE="build"
                FAILURE_STAGE="build"
                FAILURE_COMPONENT="unknown"
                echo "❌ Build failed for unknown reason"
            fi
        fi

        # Set failure details for output
        echo "failure_phase=$FAILURE_PHASE" >> $GITHUB_OUTPUT
        echo "failure_stage=$FAILURE_STAGE" >> $GITHUB_OUTPUT
        echo "failure_component=$FAILURE_COMPONENT" >> $GITHUB_OUTPUT

    else
        echo "✅ STAGE 2 PASSED: $PACKAGE_NAME package build successful"
        echo "stage2=passed" >> $GITHUB_OUTPUT
        echo "build_status=success" >> $GITHUB_OUTPUT
    fi
else
    BUILD_EXIT_CODE=$?
    echo "❌ STAGE 2 FAILED: $PACKAGE_NAME package build failed with exit code $BUILD_EXIT_CODE"
    echo "stage2=failed" >> $GITHUB_OUTPUT
    echo "build_status=failed" >> $GITHUB_OUTPUT
    exit $BUILD_EXIT_CODE
fi

echo "✅ $PACKAGE_NAME package test completed successfully on Orin!"
