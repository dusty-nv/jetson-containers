# GitHub Actions Workflow Scripts

This directory contains reusable scripts for GitHub Actions workflows in the jetson-containers project.

## Scripts

### Core Scripts
- **`step-pre-checkout-cleanup.sh`** - Cleans up permission-restricted files before checkout
- **`step-system-info.sh`** - Displays Jetson system information
- **`step-env-info.sh`** - Records comprehensive environment variables and system info
- **`step-clean-environment.sh`** - Cleans up Docker, cache, and permission-restricted directories
- **`step-git-status.sh`** - Shows git repository state and recent history

### Build Scripts
- **`step-build-package.sh <package_name>`** - Builds and tests a specific package
- **`step-analyze-results.sh`** - Analyzes build results and outputs status
- **`step-test-results-summary.sh`** - Generates a comprehensive test results summary

### Workflow Generation
- **`workflow-template.yml`** - Template for creating new package workflow files
- **`generate-workflow.sh <package_name> [platform]`** - Generates a new workflow file from template

## Usage

### Creating a New Workflow

To create a new workflow for a package and platform:

```bash
cd .github/workflows/scripts
./generate-workflow.sh vllm orin
./generate-workflow.sh vllm thor
```

This will create:
- `../workflows/pr-on-dev_vllm_orin.yml`
- `../workflows/pr-on-dev_vllm_thor.yml`

### Creating Multiple Workflows

```bash
cd .github/workflows/scripts
# For Orin platform
./generate-workflow.sh vllm orin
./generate-workflow.sh cudnn orin
./generate-workflow.sh nccl orin

# For Thor platform
./generate-workflow.sh vllm thor
./generate-workflow.sh cudnn thor
./generate-workflow.sh nccl thor
```

### Platform Support

- **`orin`** - Jetson Orin (default if platform not specified)
- **`thor`** - Jetson Thor
- **Any other platform** - Just specify the platform name

### Manual Workflow Creation

1. Copy `workflow-template.yml` to `../workflows/pr-on-dev_<package>_<platform>.yml`
2. Replace all instances of `PACKAGE_NAME` with the actual package name
3. Replace all instances of `PLATFORM` with the actual platform name
4. Update the workflow name as needed

## Script Parameters

### step-build-package.sh
- **Required**: `<package_name>` - The package to build and test
- **Example**: `./step-build-package.sh vllm`

### step-analyze-results.sh
- **Required**: `<package_name> <stage1> <stage2> <build_status>`
- **Optional**: `[failure_phase] [failure_stage] [failure_component]`

### step-test-results-summary.sh
- **Required**: `<package_name> <stage1> <stage2> <build_status>`
- **Optional**: `[failure_phase] [failure_stage] [failure_component]`

## Benefits

- **Reusable**: Same scripts work for all packages
- **Maintainable**: Update scripts once, affects all workflows
- **Consistent**: All workflows have the same behavior and output format
- **Easy to create**: Generate new workflows with a single command
