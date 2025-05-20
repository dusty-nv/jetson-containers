# Code Style Guide

This document outlines the code style standards for the jetson-containers project and how to set up automatic code formatting.

## Code Formatting Tools

The project uses several tools to maintain consistent code style:

- [black](https://github.com/psf/black): Python code formatter
- [flake8](https://flake8.pycqa.org/): Python code linter
- [pre-commit](https://pre-commit.com/): Git hooks manager

## Setup Instructions

1. Install the required tools:
   ```bash
   pip install -r requirements.txt
   ```

2. Install the pre-commit hooks:
   ```bash
   pre-commit install
   ```

3. (Optional) Run formatting on all files:
   ```bash
   pre-commit run --all-files
   ```

## How It Works

- The pre-commit hooks run automatically before each commit
- Only files in whitelisted paths are checked (see `.pre-commit-config.yaml`)
- If formatting issues are found:
  - Some will be fixed automatically
  - Others will need manual fixes
  - The commit will be blocked until all checks pass

## Adding Your Package to Formatting Checks

If you want your package to be included in the formatting checks:

1. Add your package path to the `files` patterns in `.pre-commit-config.yaml`:
   ```yaml
   files: |
       (?x)^(
           jetson_containers/.*\.py|
           packages/your-package/.*\.py|    # Add your package here
           test_precommit\.py
       )$
   ```

2. Install pre-commit (if not already installed):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Opting Out

If you don't want to use the formatting tools:
- You don't need to install pre-commit
- Your package won't be checked if it's not in the whitelist
- You can still commit and push code normally

## Troubleshooting

If you encounter issues:

1. Check pre-commit version:
   ```bash
   pre-commit --version
   ```

2. Update pre-commit:
   ```bash
   pre-commit autoupdate
   ```

3. Clear pre-commit cache:
   ```bash
   pre-commit clean
   ```

4. Run pre-commit manually:
   ```bash
   pre-commit run --all-files
   ```

## Additional Resources

- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [Pre-commit Documentation](https://pre-commit.com/)
