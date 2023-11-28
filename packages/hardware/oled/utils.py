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
    with open('/sys/devices/gpu.0/load', 'r') as f:
        return float(f.read().strip('\n')) / 1000.0

    
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