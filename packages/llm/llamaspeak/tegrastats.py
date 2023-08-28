#!/usr/bin/env python3
import os
import time
import psutil
import pprint
import threading

class Tegrastats(threading.Thread):
    """
    Reads system stats like CPU/GPU utilization, memory usage, ect.
    """
    def __init__(self, tegrastats_poll_rate=0.5, callback=None, **kwargs):
                 
        super(Tegrastats, self).__init__(daemon=True)
        
        self.gpu_path, gpu_name = self.find_gpu()
        self.poll_rate = tegrastats_poll_rate
        self.running = False
        self.callback = callback
        
        print(f"-- tegrastats GPU path:  {self.gpu_path}")
        
        cpu_freqs = psutil.cpu_freq(percpu=False)
        
        self.stats = { 
            'cpu': { 
                'cores': psutil.cpu_count(),
                'min_freq': cpu_freqs.min,
                'max_freq': cpu_freqs.max,
            },
            'memory': {
                'total': psutil.virtual_memory().total / 1048576,
            },
            'swap': {
                'total': psutil.swap_memory().total / 1048576,
            },
        }
        
        if self.gpu_path:
            self.stats['gpu'] = {
                'name': gpu_name,
                'min_freq': float(open(os.path.join(self.gpu_path, 'min_freq')).read()) / 1000000,
                'max_freq': float(open(os.path.join(self.gpu_path, 'max_freq')).read()) / 1000000,
                'available_frequencies': [float(x)/1000000 for x in open(os.path.join(self.gpu_path, 'available_frequencies')).read().split(' ')],
            }

        self.read(use_cache=False)
        pprint.pprint(self.stats)
        
    def read(self, use_cache=True):
        if self.running and use_cache:
            return self.stats
            
        cpu_freqs = psutil.cpu_freq(percpu=False)
        
        self.stats['cpu']['cur_freq'] = cpu_freqs.current
        self.stats['cpu']['load'] = psutil.cpu_percent(interval=self.poll_rate, percpu=False)
        
        virt_mem = psutil.virtual_memory()
        
        self.stats['memory']['free'] = virt_mem.available / 1048576
        self.stats['memory']['used'] = (virt_mem.total - virt_mem.available) / 1048576
        
        swap_mem = psutil.swap_memory()
        
        self.stats['swap']['free'] = swap_mem.free / 1048576
        self.stats['swap']['used'] = swap_mem.used / 1048576
        
        if self.gpu_path:
            self.stats['gpu']['load'] = float(open(os.path.join(self.gpu_path, 'device/load')).read()) / 10
            self.stats['gpu']['cur_freq'] = float(open(os.path.join(self.gpu_path, 'cur_freq')).read()) / 1000000
            
        summary = [
            f"CPU: {int(self.stats['cpu']['load'])}%@{int(self.stats['cpu']['cur_freq'])}",
            f"GPU: {int(self.stats['gpu']['load'])}%@{int(self.stats['gpu']['cur_freq'])}",
            f"Mem: {self.stats['memory']['used']/1024:.1f}/{self.stats['memory']['total']/1024:.1f}GB",
        ]
        
        self.stats['summary'] = '  '.join(summary)
        
        return self.stats
        
    def find_gpu(self, root='/sys/class/devfreq', gpus=['gv11b', 'gp10b', 'ga10b', 'gpu']):
        for item in os.listdir(root):
            item_path = os.path.join(root, item)
            if not os.path.isdir(item_path):
                continue
            for gpu in gpus:
                if '.' + gpu in item:
                    return item_path, gpu
        print("-- tegrastats warning:  couldn't find /sys/class/devfreq entry for GPU")
        return None, None
        
    def run(self):
        self.running = True
        
        while True:
            self.read(use_cache=False)
            if self.callback:
                self.callback(self.stats)
            time.sleep(self.poll_rate)
            