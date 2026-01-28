from jetson_containers import L4T_VERSION

def pybind11(version, default=False):
    pkg = package.copy()
    
    pkg['name'] = f'pybind11:{version}'
    pkg['build_args'] = {
        'PYBIND11_VERSION': version,
    }
    
    if default:
        pkg['alias'] = 'pybind11'
        
    global_pkg = pkg.copy()
    global_pkg['name'] = f'pybind11:{version}-global'
    global_pkg['alias'] = [f'pybind11:global']
    
    return pkg

package = [
    pybind11('2.13.6', default=True),
    pybind11('2.12.0'),
    pybind11('2.11.1'),
]
