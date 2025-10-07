> [!NOTE]  
> <a href="https://github.com/NVIDIA-ISAAC-ROS"><b>NVIDIA Isaac ROS</b></a><br/>
> See [`github.com/NVIDIA-ISAAC-ROS`](https://github.com/NVIDIA-ISAAC-ROS) for optimized CV/AI packages and [NITROS](https://nvidia-isaac-ros.github.io/concepts/nitros/index.html) zero-copy transport.<br/>
> For Isaac ROS containers, use the [Isaac ROS Docker Development Environment](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_common/index.html#isaac-ros-docker-development-environment).

Below are ROS/ROS2 base containers for JetPack.  These build ROS from source to run them on the needed versions of Ubuntu.

Supported ROS distros: `melodic` `noetic` `foxy` `galactic` `humble` `iron` <br/>
Supported ROS packages: `ros_base` `ros_core` `desktop`

### Installing Add-on Packages

Since the ROS distributions included in these containers are built from source, you should not install additional ROS packages for them from apt - instead these should be built from source too.  There is a helper script for this [`/ros2_install.sh`](ros2_install.sh) which takes either a list of ROS package names or URL of a git repo, and builds/installs them in a ROS workspace:

```
# adds foxglove to ROS_ROOT (under /opt/ros)
/ros2_install.sh foxglove_bridge

# adds jetson-inference nodes under /ros2_workspace
ROS_WORKSPACE=/ros2_workspace /ros2_install.sh https://github.com/dusty-nv/ros_deep_learning
```

You can run this from the ROS2 container using a mounted directory for your workspace (where your compiled packages will be saved outside container), or via another Dockerfile using the ROS2 container as base (in which case your packages will be built into the container itself):

* [`/packages/robots/ros/Dockerfile.ros2.extras`](/packages/robots/ros/Dockerfile.ros2.extras)
* [`/packages/jetson-inference/Dockerfile.ros`](/packages/jetson-inference/Dockerfile.ros)

Examples of this being done you can find in the [`ros:humble-foxglove`](/packages/ros) and [`jetson-inference:humble`](/packages/jetson-inference) containers.

