import copy

from jetson_containers import L4T_VERSION
from .version import ROS_DISTROS, ROS2_DISTROS, ROS_PACKAGES

# add permutations of ROS distros/packages as subpackages
template = package.copy()

template['group'] = 'ros'
template['depends'] = ['cuda', 'cudnn', 'tensorrt', 'opencv', 'cmake']
template['postfix'] = f"l4t-r{L4T_VERSION}"
template['docs'] = "docs.md"

package = []

for ROS_DISTRO in ROS_DISTROS:
    for ROS_PACKAGE in ROS_PACKAGES:
        pkg = copy.deepcopy(template)
        
        pkg['name'] = f"ros:{ROS_DISTRO}-{ROS_PACKAGE.replace('_', '-')}"
        
        pkg['build_args'] = {
            'ROS_VERSION': ROS_DISTRO,
            'ROS_PACKAGE': ROS_PACKAGE
        }
        
        if ROS_DISTRO == 'melodic':
            pkg['dockerfile'] = 'Dockerfile.ros.melodic'
            pkg['test'] = 'test_ros.sh'
            pkg['requires'] = '<34'   # melodic is for 18.04 only
            pkg['notes'] = 'ROS Melodic is for JetPack 4 only'
            pkg['depends'].remove('opencv')  # melodic apt packages install the ubuntu opencv
            pkg['depends'][0] = 'cmake:apt'  # melodic (python 2.7) doesn't like pip-based cmake
        elif ROS_DISTRO == 'noetic':
            pkg['dockerfile'] = 'Dockerfile.ros.noetic'
            pkg['test'] = 'test_ros.sh'
        else:
            pkg['dockerfile'] = 'Dockerfile.ros2'
            pkg['test'] = 'test_ros2.sh'
            
        package.append(pkg)
        
def ros_container(name, *packages, distros=ROS2_DISTROS, base_packages='desktop'):
    if not isinstance(distros, (list, tuple)):
        distros = [distros]
        
    if not isinstance(packages, (list, tuple)):
        packages = [packages]
      
    if not isinstance(base_packages, (list, tuple)):
        base_packages = [base_packages]
          
    packages = ' '.join(packages)

    if not packages:
        return
        
    for distro in distros:
        for base_package in base_packages:
            pkg = template.copy()
            
            pkg['name'] = f'ros:{distro}-{name}'
            pkg['dockerfile'] = 'Dockerfile.ros2.extras'
            pkg['depends'] = f"ros:{distro}-{base_package}"
            pkg['build_args'] = {'ROS_PACKAGES': f'{packages}'}
            
            package.append(pkg)
        
ros_container('foxglove', 'foxglove_bridge')
