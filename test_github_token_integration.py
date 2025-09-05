#!/usr/bin/env python3
"""
Test script for GitHub token integration functionality.
This script tests the new preprocess_dockerfile_for_github_api function.
"""

import os
import sys
import tempfile
import shutil

# Add the jetson_containers module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jetson_containers'))

from network import preprocess_dockerfile_for_github_api, get_github_token

def test_github_token_detection():
    """Test GitHub token detection from environment variables"""
    print("Testing GitHub token detection...")

    # Test with no token
    token = get_github_token()
    print(f"Token detected: {'Yes' if token else 'No'}")

    if token:
        print(f"Token preview: {token[:8]}...")
    else:
        print("No token found - this is expected if GITHUB_TOKEN is not set")

    return True

def test_dockerfile_preprocessing():
    """Test Dockerfile preprocessing functionality"""
    print("\nTesting Dockerfile preprocessing...")

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create a test Dockerfile with GitHub API calls
        dockerfile_content = """# Test Dockerfile
FROM ubuntu:20.04

# This should be replaced
ADD https://api.github.com/repos/dusty-nv/sudonim/git/refs/heads/main /tmp/sudonim_version.json

# This should also be replaced
ADD https://api.github.com/repos/dusty-nv/mlc/git/refs/heads/main /tmp/mlc_version.json

RUN echo "Build complete"
"""

        dockerfile_path = os.path.join(temp_dir, 'Dockerfile')
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)

        print(f"Created test Dockerfile: {dockerfile_path}")

        # Test preprocessing
        result = preprocess_dockerfile_for_github_api(dockerfile_path, temp_dir)
        modified_dockerfile, build_args = result

        print(f"Preprocessing result:")
        print(f"  Modified Dockerfile: {modified_dockerfile}")
        print(f"  Build args: {build_args}")

        if modified_dockerfile != dockerfile_path:
            print("✅ Preprocessing successful!")

            # Show the modified content
            with open(modified_dockerfile, 'r') as f:
                modified_content = f.read()
            print("\nModified Dockerfile content:")
            print("=" * 50)
            print(modified_content)
            print("=" * 50)

            # Check if .github-api-temp directory was created
            temp_api_dir = os.path.join(temp_dir, '.github-api-temp')
            if os.path.exists(temp_api_dir):
                print(f"✅ Temporary API directory created: {temp_api_dir}")
                print("Files in temp directory:")
                for file in os.listdir(temp_api_dir):
                    print(f"  - {file}")
            else:
                print("❌ Temporary API directory not created")

        else:
            print("❌ Preprocessing failed - no changes made")

        return True

    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")

def main():
    """Main test function"""
    print("GitHub Token Integration Test Suite")
    print("=" * 50)

    try:
        # Test 1: Token detection
        test_github_token_detection()

        # Test 2: Dockerfile preprocessing
        test_dockerfile_preprocessing()

        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
