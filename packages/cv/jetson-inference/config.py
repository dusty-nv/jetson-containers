from ..robots.ros.version import ROS2_DISTROS

package['name'] = 'jetson-inference:main'
package['alias'] = 'jetson-inference'

package = [package]

for distro in ROS2_DISTROS:
    ros = package[0].copy()
 
    ros['name'] = f'jetson-inference:{distro}'
    ros['depends'] = [f'ros:{distro}-desktop', 'jetson-inference:main']
    ros['dockerfile'] = 'Dockerfile.ros'
    ros['test'] = ros['test'] + ['test_ros.sh']
    
    del ros['alias']
    package.append(ros)

