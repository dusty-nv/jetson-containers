from jetson_containers import PYTHON_VERSION, PYTHON_FREE_THREADING
from packaging.version import Version

def python(version, requires=None) -> list:
    pkg = package.copy()

    pkg['name'] = f'python:{version}'
    
    # Detect if this is a free-threaded build
    is_free_threaded = version.endswith('t') or '-nogil' in version
    version_base = version.rstrip('t').rstrip('-nogil')
    
    # Pass clean version and free-threading flag separately
    pkg['build_args'] = {
        'PYTHON_VERSION': version_base,
        'PYTHON_FREE_THREADING': '1' if is_free_threaded else '0'
    }

    # Set as default if version matches PYTHON_VERSION
    try:
        if Version(version_base) == PYTHON_VERSION:
            # Also check if free-threading matches
            if is_free_threaded == PYTHON_FREE_THREADING:
                pkg['alias'] = 'python'
    except Exception:
        pass  # Skip alias for non-standard versions

    if requires:
        pkg['requires'] = requires

    return pkg

package = [
    python('3.6', '==32.*'),  # JetPack 4
    python('3.8', '<36'),     # JetPack 4 + 5
    python('3.10', '>=34'),   # JetPack 5 + 6
    python('3.11', '>=34'),   # JetPack 6
    python('3.12', '>=34'),   # JetPack 6
    python('3.13', '>=34'),   # JetPack 6
    python('3.14', '>=34'),   # JetPack 6
    python('3.14t', '>=34'),  # JetPack 6 - Free-threaded (no-GIL)
]
