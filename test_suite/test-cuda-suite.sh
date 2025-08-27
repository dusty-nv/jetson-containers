#!/bin/bash

#==============================================================================
#                             CONFIGURATION
#==============================================================================

# --- Title for the final summary report ---
SUMMARY_TITLE="CUDA Packages Update"

# --- Environment details for the summary report ---
SUMMARY_INFO="Cu126, py310, lts2204"

# --- Directory containing the packages to be built ---
PACKAGES_DIR="./packages/cuda"

#==============================================================================
#                         SCRIPT LOGIC
#==============================================================================

# --- Log file for detailed output ---
LOG_FILE="build_summary.log"
echo "Starting build process at $(date)" > "$LOG_FILE"
echo "-------------------------------------" >> "$LOG_FILE"

# --- Associative arrays and lists ---
declare -A BUILD_RESULTS
declare -A FAILURE_NOTES
declare -A PACKAGE_PATHS
declare -A DEPENDENCIES
PACKAGES_TO_BUILD=()

# --- Step 1: Discover all packages and their declared dependencies ---
echo -e "\n[-->] Discovering packages and dependencies from '$PACKAGES_DIR'..."

if [ ! -d "$PACKAGES_DIR" ]; then
    echo "[!!!] Error: Packages directory '$PACKAGES_DIR' not found. Exiting."
    exit 1
fi

for dir in $PACKAGES_DIR/*/; do
    if [ -d "$dir" ]; then
        pkg=$(basename "$dir")
        PACKAGES_TO_BUILD+=("$pkg")
        PACKAGE_PATHS["$pkg"]="$dir"
        
        dockerfile="${dir}Dockerfile"
        if [ -f "$dockerfile" ]; then
            dependency_line=$(grep -m 1 "# depends:" "$dockerfile")
            # Only add a dependency if the line is found and is not empty
            if [[ -n "$dependency_line" ]]; then
                dependency=$(echo "$dependency_line" | sed -n 's/.*# depends: \[\(.*\)\]/\1/p' | awk -F, '{print $1}')
                if [[ -n "$dependency" ]]; then
                    DEPENDENCIES["$pkg"]="$dependency"
                    echo "      - Found package: $pkg (depends on '$dependency')"
                else
                    echo "      - Found package: $pkg (no dependency specified)"
                fi
            else
                 echo "      - Found package: $pkg (no '# depends:' line found)"
            fi
        else
            echo "[!!!] Warning: Dockerfile not found for package '$pkg' in '$dir'."
        fi
    fi
done
echo "[OK] Package discovery complete."


# --- Step 2: Determine build order with robust cycle detection and handling of external deps ---
echo -e "\n[-->] Determining build order..."
declare -a SORTED_PACKAGES
declare -A VISITING_PATH # Stores the current path to detect cycles
declare -A VISITED # Stores nodes that have been fully processed

function visit() {
    local pkg=$1
    
    VISITED[$pkg]=1
    VISITING_PATH[$pkg]=1
    
    local dependency=${DEPENDENCIES[$pkg]}
    
    # Only process a dependency if it's declared for this package
    if [[ -n "$dependency" ]]; then
        # Check if the dependency is another local package we need to build
        if [[ " ${PACKAGES_TO_BUILD[*]} " =~ " ${dependency} " ]]; then
            # Check for a circular dependency among local packages
            if [[ -n "${VISITING_PATH[$dependency]}" ]]; then
                echo "[!!!] Validation Error: Circular Dependency Detected!"
                echo "      - The build path created a loop."
                echo "      - Path: $(IFS=" -> "; echo "${!VISITING_PATH[*]}") -> $dependency"
                exit 1
            fi
            
            # If the local dependency hasn't been visited yet, visit it first
            if [[ -z "${VISITED[$dependency]}" ]]; then
                visit "$dependency"
            fi
        else
            # This is an external dependency (e.g., 'build-essential', 'numpy')
            echo "      - Info: Package '$pkg' has an external dependency '$dependency', assuming it is met by the system."
        fi
    fi
    
    # Add the package to the sorted list after its dependencies are handled
    unset VISITING_PATH[$pkg]
    SORTED_PACKAGES+=("$pkg")
}

for pkg in "${PACKAGES_TO_BUILD[@]}"; do
    if [[ -z "${VISITED[$pkg]}" ]]; then
        visit "$pkg"
    fi
done

echo "[OK] Build order determined: ${SORTED_PACKAGES[*]}"


# --- Main build loop (No changes below this line) ---
for pkg in "${SORTED_PACKAGES[@]}"; do
    echo -e "\n[-->] Processing package: $pkg"
    
    PKG_DIR=${PACKAGE_PATHS[$pkg]}

    dependency=${DEPENDENCIES[$pkg]}
    # Only check for failure of LOCAL dependencies
    if [[ -n "$dependency" && " ${PACKAGES_TO_BUILD[*]} " =~ " ${dependency} " && "${BUILD_RESULTS[$dependency]}" == "fail" ]]; then
        echo "[!!!] Skipping $pkg because local dependency '$dependency' failed to build."
        BUILD_RESULTS[$pkg]="fail"
        FAILURE_NOTES[$pkg]="(due to $dependency failure)"
        continue
    fi

    BUILD_COMMAND="jetson-containers build --name test-${pkg} ${pkg}"
    echo "[$] Running command: $BUILD_COMMAND"
    
    output=$(eval "$BUILD_COMMAND" 2>&1)
    exit_code=$?
    clean_output=$(echo "$output" | tr -d '\000')

    echo "--- Output for $pkg ---" >> "$LOG_FILE"
    echo "$output" >> "$LOG_FILE"
    echo "--- End of output for $pkg (Exit Code: $exit_code) ---" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    if [ $exit_code -eq 0 ] && echo "$clean_output" | grep -q "Built test-${pkg}"; then
        echo "[OK] ✅ Successfully built $pkg"
        BUILD_RESULTS[$pkg]="success"
    else
        echo "[FAIL] ❌ Failed to build $pkg. See $LOG_FILE for details."
        BUILD_RESULTS[$pkg]="fail"
        
        if echo "$clean_output" | grep -iq "permission denied"; then
            FAILURE_NOTES[$pkg]="(permission denied)"
        elif [ $exit_code -ne 0 ]; then
             FAILURE_NOTES[$pkg]="(non-zero exit code: $exit_code)"
        else
            FAILURE_NOTES[$pkg]="(build process failed)"
        fi
    fi
done

# --- Generate Final Summary ---
echo -e "\n\n=============================================="
echo "          BUILD SUMMARY"
echo "=============================================="
echo -e "\n${SUMMARY_TITLE}\n"
echo "${SUMMARY_INFO}"

for pkg in "${SORTED_PACKAGES[@]}"; do
    display_name="$(tr '[:lower:]' '[:upper:]' <<< ${pkg:0:1})${pkg:1}"

    if [[ "${BUILD_RESULTS[$pkg]}" == "success" ]]; then
        echo "${display_name} ✅"
    else
        reason=${FAILURE_NOTES[$pkg]}
        echo "${display_name} ⛓️‍💥 ${reason}"
    fi
done

echo -e "\nFull build logs are available in: ${LOG_FILE}"
echo "=============================================="