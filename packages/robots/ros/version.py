
from jetson_containers import LSB_RELEASE, log_warning
import os

ROS1_DISTROS = ['melodic', 'noetic']
ROS2_DISTROS = ['foxy', 'galactic', 'humble', 'iron', 'jazzy']

ROS_DISTROS = ROS1_DISTROS + ROS2_DISTROS
ROS_PACKAGES = ['ros_base', 'ros_core', 'desktop']

if 'ROS_DISTRO' in os.environ:
  ROS_DISTRO = os.environ.get('ROS_DISTRO')
if LSB_RELEASE == '24.04':
  ROS_DISTRO = 'jazzy'
elif LSB_RELEASE == '22.04':
  ROS_DISTRO = 'humble'
elif LSB_RELEASE == '20.04':
  ROS_DISTRO = 'noetic'
elif LSB_RELEASE == '18.04':
  ROS_DISTRO = 'melodic'
else:
  ROS_DISTRO = 'humble'
  log_warning(f"defaulting to ROS_DISTRO={ROS_DISTRO} after unrecognized LSB_RELEASE={LSB_RELEASE}")

__all__ = ['ROS_DISTROS', 'ROS1_DISTROS', 'ROS2_DISTROS', 'ROS_PACKAGES']
