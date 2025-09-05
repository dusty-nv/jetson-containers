#!/usr/bin/env python3
"""
Simple test script to verify webhook functionality
"""

import os
import sys
import tempfile
from datetime import datetime

# Add the current directory to the Python path so we can import our modules
sys.path.insert(0, '/home/thor/git/jetson-containers')

# Mock the missing dependencies for testing
class MockModule:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

sys.modules['wget'] = MockModule()
sys.modules['dockerhub_api'] = MockModule()
sys.modules['tabulate'] = MockModule()
sys.modules['termcolor'] = MockModule()

# Now import our webhook functions
from jetson_containers.network import send_webhook, get_log_tail

def test_webhook_success():
    """Test webhook with success status"""
    print("Testing webhook with success status...")
    
    # Set webhook URL for testing
    os.environ['WEBHOOK_URL'] = 'https://httpbin.org/post'
    
    packages = ['pytorch', 'ros:humble']
    message = "Successfully built packages: pytorch, ros:humble"
    
    try:
        send_webhook('success', packages, message)
        print("✅ Success webhook test completed")
    except Exception as e:
        print(f"❌ Success webhook test failed: {e}")

def test_webhook_failure():
    """Test webhook with failure status and log tail"""
    print("Testing webhook with failure status...")
    
    # Create a temporary log file for testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        log_content = [
            "Starting build process...",
            "Downloading packages...", 
            "Building container...",
            "Running tests...",
            "ERROR: Test failed",
            "ERROR: Container build failed",
            "ERROR: Unable to complete build",
            "Build process terminated",
            "Exit code: 1",
            "End of build log"
        ]
        f.write('\n'.join(log_content))
        log_file_path = f.name
    
    try:
        # Test log tail functionality
        tail = get_log_tail(log_file_path, 5)
        print(f"Log tail (last 5 lines):\n{tail}")
        
        # Test webhook with failure message including log tail
        packages = ['pytorch']
        message = f"Build failed for packages: pytorch\nError: Test failure\n\nLast 5 lines from build log:\n{tail}"
        
        send_webhook('failure', packages, message)
        print("✅ Failure webhook test completed")
        
    except Exception as e:
        print(f"❌ Failure webhook test failed: {e}")
    finally:
        # Clean up temp file
        os.unlink(log_file_path)

def test_no_webhook_url():
    """Test that webhook is skipped when WEBHOOK_URL is not set"""
    print("Testing webhook without WEBHOOK_URL...")
    
    # Remove webhook URL
    if 'WEBHOOK_URL' in os.environ:
        del os.environ['WEBHOOK_URL']
    
    try:
        send_webhook('success', ['test'], 'Test message')
        print("✅ No webhook URL test completed (should skip silently)")
    except Exception as e:
        print(f"❌ No webhook URL test failed: {e}")

if __name__ == '__main__':
    print("Webhook Implementation Test")
    print("=" * 50)
    
    test_no_webhook_url()
    print()
    test_webhook_success()
    print()
    test_webhook_failure()
    
    print("\nWebhook implementation test completed!")
