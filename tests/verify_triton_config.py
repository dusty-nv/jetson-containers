
import sys
import os
import unittest
from unittest.mock import MagicMock
import importlib

# Mock the jetson_containers module
sys.modules['jetson_containers'] = MagicMock()
from packaging.version import Version

def mock_L4T_VERSION(version_str):
    sys.modules['jetson_containers'].L4T_VERSION = Version(version_str)

def mock_LSB_RELEASE(release_str):
    sys.modules['jetson_containers'].LSB_RELEASE = release_str

class TestTritonConfig(unittest.TestCase):
    def setUp(self):
        # Allow reloading config.py to pick up new mock values
        if 'packages.ml.tritonserver.config' in sys.modules:
            del sys.modules['packages.ml.tritonserver.config']
        # Also need to make sure we can import from the path relative to where we run the test
        sys.path.append('.')


    def run_config(self):
        # Read the config file
        with open('packages/ml/tritonserver/config.py', 'r') as f:
            lines = f.readlines()
        
        # Filter out imports to rely on context
        filtered_lines = [line for line in lines if not line.strip().startswith('from ') and not line.strip().startswith('import ')]
        content = 'print(f"DEBUG: L4T_VERSION={L4T_VERSION}, type={type(L4T_VERSION)}")\n' + \
                  'print(f"DEBUG: Comparison Result={L4T_VERSION >= Version(\'36.4.0\')}")\n' + \
                  ''.join(filtered_lines)
        
        # Prepare context with the mocked package dict
        context = {
            'package': {'name': 'tritonserver'},
            'os': os,
            'L4T_VERSION': sys.modules['jetson_containers'].L4T_VERSION,
            'Version': Version,
            'LSB_RELEASE': sys.modules['jetson_containers'].LSB_RELEASE
        }
        
        # Execute the config
        exec(content, context)
        return context

    def test_l4t_36_4_0_ubuntu_22_04(self):
        mock_L4T_VERSION('36.4.0')
        mock_LSB_RELEASE('22.04')
        
        context = self.run_config()
        
        TRITON_URL = context['package']['build_args']['TRITON_URL']
        TRITON_TAR = context['package']['build_args']['TRITON_TAR']
        TRITON_VERSION = context['package']['build_args']['TRITON_VERSION']
        
        self.assertEqual(TRITON_VERSION, '2.49.0')
        self.assertIn('v2.49.0', TRITON_URL)
        self.assertTrue(TRITON_TAR.endswith('.tar.gz'))
        self.assertIn('vllm', context['package']['depends'])
        self.assertIn('numpy', context['package']['depends'])

    def test_l4t_36_4_0_ubuntu_24_04(self):
        mock_L4T_VERSION('36.4.0')
        mock_LSB_RELEASE('24.04')
        
        context = self.run_config()
        
        TRITON_URL = context['package']['build_args']['TRITON_URL']
        TRITON_TAR = context['package']['build_args']['TRITON_TAR']
        TRITON_VERSION = context['package']['build_args']['TRITON_VERSION']

        self.assertEqual(TRITON_VERSION, '2.63.0')
        self.assertIn('v2.63.0', TRITON_URL)
        self.assertTrue(TRITON_TAR.endswith('.tar.gz'))

    def test_l4t_36_0_0(self):
        mock_L4T_VERSION('36.0.0')
        mock_LSB_RELEASE('22.04') 
        
        context = self.run_config()
        
        TRITON_URL = context['package']['build_args']['TRITON_URL']
        self.assertIn('v2.59.1', TRITON_URL)

if __name__ == '__main__':
    unittest.main()
