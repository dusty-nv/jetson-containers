from .version import ROS_DISTRO, ROS2_DISTROS
import copy

def ros_container(package: dict, *sources: str, 
                  distros: list=ROS2_DISTROS, 
                  base_packages: list=['desktop'], 
                  workspace: str='/workspace', 
                  dockerfile: str=None,
                  depends: list=[], tag: str=None, 
                  **kwargs):
    """
    Define a set of containers from permuting the ROS distros and base packages,
    along with the ROS packages from the given list of `sources` to build.
    
    The list of ROS sources can either be URLs to GitHub repos that `catkin` would 
    build, or as the names of ROS packages in the distribution (like `vision_msgs`)
    To clone a specific git branch, use `https://github.com/user/repo@branch`

    Inside the container, it uses `/ros2_install.sh` which remains there to use.
    The packages will get installed under the given `workspace` and automatically
    sourced into the environment.  See here to use this from other config.py files:

        from ..robots.ros import ros_container

        # add some example ROS packages to humble:desktop and jazzy:desktop containers
        package = ros_container(package, 'my_ros_pkg', 'https://github.com/ros/package',
                                distro=['humble', 'jazzy'], base_packages='desktop')

    This takes one package configuration in `package` and returns a list of them.
    The kwargs are set as attributes on the list of packages, merged into each dict.
    Set the `name` kwarg with variables like `${ROS_DISTRO}` for naming containers.
    """
    if not isinstance(distros, (list, tuple)):
        distros = [distros]
        
    if not isinstance(sources, (list, tuple)):
        sources = [sources]
      
    if not isinstance(base_packages, (list, tuple)):
        base_packages = [base_packages]

    packages = []  

    for k,v in kwargs.items():
        package[k] = v

    if ':' not in package['name']:
        package['name'] += ':${ROS_DISTRO}-${ROS_PACKAGE}-${TAG}'

    if not dockerfile:
        dockerfile = '/'.join(__file__.split('/')[:-1]) + '/Dockerfile.ros2.extras'
        
    for distro in distros:
        for base_package in base_packages:
            for source in sources:
                pkg = copy.deepcopy(package)

                if distro == ROS_DISTRO and 'alias' not in pkg:
                    pkg['alias'] = pkg['name'].split(':')[0] + ':$TAG'

                pkg['dockerfile'] = dockerfile
                pkg['depends'] = [f'ros:{distro}-{base_package}', 'vpi'] + depends

                pkg['build_args'] = {
                    'ROS_PACKAGE': f'{source}',
                    'ROS_WORKSPACE': workspace,
                }

                if 'build_args' in kwargs:
                    pkg['build_args'].merge(kwargs['build_args'])

                subs = dict(
                    ROS_DISTRO=distro, ROS_PACKAGE=base_package,
                    TAG=tag if tag else source.split('/')[-1].split('@')[0]
                )

                for k,v in subs.items():
                    a = '$'+k
                    b = '${'+k+'}'
                    pkg['name'] = pkg['name'].replace(a,b).replace(b,v)
                    if 'alias' in pkg:
                        pkg['alias'] = pkg['alias'].replace(a,b).replace(b,v) 
                    for i,d in enumerate(pkg['depends']):
                        pkg['depends'][i] = d.replace(a,b).replace(b,v)

                if 'http' in source:
                    a = source.find('@')
                    if a >= 0:
                        pkg['build_args'].merge({
                            'ROS_PACKAGE': source[:a],
                            'ROS_BRANCH': source[a+1:],
                        })

                packages.append(pkg)

    return packages
        

__all__ = ['ros_container']
