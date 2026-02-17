from jetson_containers import L4T_VERSION, LSB_RELEASE
from ..robots.ros import ros_container, ROS_DISTROS

ISAAC_ROS_URL="https://github.com/NVIDIA-ISAAC-ROS"

def isaac_ros(repo, version='4.0', base_url=ISAAC_ROS_URL, workspace='/opt/isaac-ros', depends=[], tag=None, **kwargs):
    """
    Generate a container build with an Isaac ROS package added to it,
    which by default is under `/opt/isaac-ros` and sourced on startup.
    """
    if repo != 'isaac_ros_common':
        depends.insert(0, f'isaac-ros:common-{version}-$ROS_DISTRO-$ROS_PACKAGE')

    if not tag:
        prefix='isaac_ros_'
        if repo.startswith(prefix):
            tag=repo[len(prefix):].replace('_', '-')

    for idx, dep in enumerate(depends):
        if dep.startswith('isaac-ros:') and not any([x in dep for x in ROS_DISTROS + ['ROS_DISTRO']]):
            depends[idx] = dep + f"-{version}-$ROS_DISTRO-$ROS_PACKAGE"

    kwargs.setdefault('name', f'isaac-ros:$TAG-{version}-$ROS_DISTRO-$ROS_PACKAGE')
    kwargs.setdefault('group', 'robots')
    kwargs.setdefault('distros', ['humble', 'jazzy'])
    kwargs.setdefault('test', ['test.sh'])

    return ros_container(package, f"{base_url}/{repo}", depends=depends, tag=tag, workspace=workspace, **kwargs)

package = [
    isaac_ros('isaac_ros_common', dockerfile='Dockerfile'),
    isaac_ros('isaac_ros_nitros', depends=['cuda-python', 'cv-cuda:cpp']),
    isaac_ros('isaac_ros_image_pipeline', depends=['isaac-ros:nitros', 'ffmpeg:git']),
    isaac_ros('isaac_ros_dnn_inference', depends=['isaac-ros:image-pipeline']),
    isaac_ros('isaac_ros_compression', depends=['isaac-ros:image-pipeline']),
    isaac_ros('isaac_ros_visual_slam', depends=['isaac-ros:compression']),
    isaac_ros('isaac_ros_pose_estimation', depends=['isaac-ros:compression']),
    isaac_ros('isaac_ros_nvblox', depends=['isaac-ros:nitros']),
    isaac_ros('isaac_manipulator', depends=['isaac-ros:nvblox'], tag='manipulator'),
    isaac_ros('isaac_ros_data_tools'),
]
