import subprocess
import pkg_resources
import platform
import os

def notebooks_dir():
    return pkg_resources.resource_filename('jetbot', 'notebooks')


def platform_notebooks_dir():
    if 'aarch64' in platform.machine():
        return os.path.join(notebooks_dir(), 'robot')
    else:
        return os.path.join(notebooks_dir(), 'host')
    

def platform_model_str():
    with open('/proc/device-tree/model', 'r') as f:
        return str(f.read()[:-1])


def platform_is_nano():
    return 'jetson-nano' in platform_model_str()


def ip_address(interface):
    try:
        if network_interface_state(interface) == 'down':
            return None
        cmd = "hostname -I | cut -d' ' -f1"
        return subprocess.check_output(cmd, shell=True).decode('ascii')[:-1]
    except:
        return None


def network_interface_state(interface):
    try:
        with open('/sys/class/net/%s/operstate' % interface, 'r') as f:
            return f.read()
    except:
        return 'down' # default to down

        
def power_mode():
    """Gets the Jetson's current power mode
    
    Gets the current power mode as set by the tool ``nvpmodel``.
    
    Returns:
        str: The current power mode.  Either 'MAXN' or '5W'.
    """
    #return subprocess.check_output("nvpmodel -q | grep -o '5W\|MAXN'", shell = True ).decode('utf-8').strip('\n')
    return "Max-N"

def power_usage():
    """Gets the Jetson's current power usage in Watts
    
    Returns:
        float: The current power usage in Watts.
    """
    #with open("/sys/devices/50000000.host1x/546c0000.i2c/i2c-6/6-0040/iio:device0/in_power0_input", 'r') as f:
        #return float(f.read()) / 1000.0
    return 0.0

    
def cpu_usage():
    """Gets the Jetson's current CPU usage fraction
    
    Returns:
        float: The current CPU usage fraction.
    """
    return float(subprocess.check_output("top -bn1 | grep load | awk '{printf \"%.2f\", $(NF-2)}'", shell = True ).decode('utf-8'))


def gpu_usage():
    """Gets the Jetson's current GPU usage fraction
    
    Returns:
        float: The current GPU usage fraction.
    """
    load = 1
    gpu_path = "/sys/class/devfreq/"
    extensionsToCheck = ('.gv11b', '.gp10b', '.ga10b', '.gpu')
    for item in os.listdir(gpu_path):
        if item.endswith(extensionsToCheck):
            path = os.path.realpath(os.path.join(gpu_path, item, "device"))

    #gpu_load_path = "/sys/class/devfreq/17000000.ga10b/device/load"
    gpu_load_path = os.path.realpath(os.path.join(path, "load"))
    
    with open(gpu_load_path, 'r') as f:
        # Read current GPU load
        load = float(f.read()) / 1000.0
    return load

    
def memory_usage():
    """Gets the Jetson's current RAM memory usage fraction
    
    Returns:
        float: The current RAM usage fraction.
    """
    return float(subprocess.check_output("free -m | awk 'NR==2{printf \"%.2f\", $3*100/$2 }'", shell = True ).decode('utf-8')) / 100.0


def disk_usage():
    """Gets the Jetson's current disk memory usage fraction
    
    Returns:
        float: The current disk usage fraction.
    """
    return float(subprocess.check_output("df -h | awk '$NF==\"/\"{printf \"%s\", $5}'", shell = True ).decode('utf-8').strip('%')) / 100.0

def cat(path):
    with open(path, 'r') as f:
        return f.readline().rstrip('\x00')

def find_igpu(igpu_path):
    # Check if exist a integrated gpu
    # if not os.path.exists("/dev/nvhost-gpu") and not os.path.exists("/dev/nvhost-power-gpu"):
    #     return []
    igpu = {}
    if not os.path.isdir(igpu_path):
        return igpu
    for item in os.listdir(igpu_path):
        item_path = os.path.join(igpu_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            # Check name device
            name_path = "{item}/device/of_node/name".format(item=item_path)
            if os.path.isfile(name_path):
                # Decode name
                name = cat(name_path)
                # Check if gpu
                if name in ['gv11b', 'gp10b', 'ga10b', 'gpu']:
                    # Extract real path GPU device
                    path = os.path.realpath(os.path.join(item_path, "device"))
                    frq_path = os.path.realpath(item_path)
                    igpu[name] = {'type': 'integrated', 'path': path, 'frq_path': frq_path}
                    # Check if railgate exist
                    path_railgate = os.path.join(path, "railgate_enable")
                    if os.path.isfile(path_railgate):
                        igpu[name]['railgate'] = path_railgate
                    # Check if 3d scaling exist
                    path_3d_scaling = os.path.join(path, "enable_3d_scaling")
                    if os.path.isfile(path_3d_scaling):
                        igpu[name]['3d_scaling'] = path_3d_scaling
    return igpu
