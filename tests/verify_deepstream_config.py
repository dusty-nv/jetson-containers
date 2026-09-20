import sys
import types
import unittest
from pathlib import Path

from packaging.version import Version


CONFIG = Path(__file__).parents[1] / 'packages/cv/deepstream/config.py'


def run_config(l4t_version):
    module = types.ModuleType('jetson_containers')
    module.L4T_VERSION = Version(l4t_version)
    module.PYTHON_VERSION = Version('3.10')
    previous = sys.modules.get('jetson_containers')
    sys.modules['jetson_containers'] = module
    try:
        package = {'name': 'deepstream'}
        context = {'package': package}
        exec(compile(CONFIG.read_text(), str(CONFIG), 'exec'), context)
        return context['package']
    finally:
        if previous is None:
            del sys.modules['jetson_containers']
        else:
            sys.modules['jetson_containers'] = previous


class TestDeepStreamConfig(unittest.TestCase):
    def test_supported_resolution_matrix(self):
        cases = (
            ('32.6.0', '6.0.0', '1.1.1', 'nvidia.box.com'),
            ('35.2.1', '6.3.0', '1.1.8', '/deepstream/6.3/'),
            ('36.2.0', '6.4.0', '1.1.10', '/deepstream/6.4/'),
            ('36.4.2', '7.1.0', '1.2.0', '/deepstream/7.1/'),
            ('36.4.3', '7.1.0', '1.2.0', '/deepstream/7.1/'),
            ('36.4.7', '7.1.0', '1.2.0', '/deepstream/7.1/'),
            ('38.2.0', '8.0.0', '1.2.2', '/deepstream/8.0/'),
        )
        for l4t, deepstream, pyds, url_marker in cases:
            with self.subTest(l4t=l4t):
                build_args = run_config(l4t)['build_args']
                self.assertIn(url_marker, build_args['DEEPSTREAM_URL'])
                self.assertEqual(build_args['DEEPSTREAM_TAR'], f'deepstream_sdk_v{deepstream}_jetson.tbz2')
                self.assertEqual(build_args['PYDS_VERSION'], pyds)

    def test_unsupported_newer_releases_fail_closed(self):
        for l4t in ('38.4.0', '39.2.0', '40.0.0'):
            with self.subTest(l4t=l4t):
                self.assertIsNone(run_config(l4t))


if __name__ == '__main__':
    unittest.main()
