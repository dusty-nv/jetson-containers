
from jetson_containers import L4T_VERSION

def NanoLLM(version, branch=None, requires=None, default=False, ros=['foxy', 'galactic', 'humble', 'iron', 'jazzy']):
    pkg = package.copy()

    pkg['name'] = f"nano_llm:{version}"

    if default:
        pkg['alias'] = 'nano_llm'

    if requires:
        pkg['requires'] = requires

    if L4T_VERSION.major >= 36:
        pkg['depends'] = ['awq'] + pkg['depends'] + ['whisper_trt']

    if not branch:
        branch = version

    pkg['build_args'] = {'NANO_LLM_BRANCH': branch}

    if not isinstance(ros, (list, tuple)):
        ros = [ros]

    containers = [pkg]

    for ros_distro in ros:
        r = pkg.copy()

        r['name'] = f'nano_llm:{version}-{ros_distro}'
        r['depends'] = [f'ros:{ros_distro}-desktop'] + [f'jetson-inference:{ros_distro}'] + [pkg['name']]
        r['dockerfile'] = 'Dockerfile.ros'
        r['test'] = 'test_ros.sh'

        if default:
            r['alias'] = f'nano_llm:{ros_distro}'

        containers.append(r)

    return containers

package = [
    NanoLLM('main', default=True),
    NanoLLM('24.4'),
    NanoLLM('24.4.1'),
    NanoLLM('24.5'),
    NanoLLM('24.5.1'),
    NanoLLM('24.6'),
    NanoLLM('24.7'),
]


