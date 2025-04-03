
def habitat_sim(version, branch=None, default=False):
    pkg = package.copy()
    pkg['name'] = f'habitat-sim:{version}'

    if not branch:
        branch = f'v{version}'
        
    pkg['build_args'] = {
        'HABITAT_SIM_BRANCH': branch,
        'HABITAT_SIM_VERSION': version,
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'habitat-sim:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'habitat-sim'
        builder['alias'] = 'habitat-sim:builder'

    return pkg, builder

package = [
    habitat_sim('0.3.4', default=True),
]
