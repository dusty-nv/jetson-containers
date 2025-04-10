import threading
import Adafruit_SSD1306
import time
import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw
from flask import Flask
from utils import ip_address, power_mode, power_usage, cpu_usage, gpu_usage, memory_usage, disk_usage

from smbus2 import SMBus
import logging

logger = logging.getLogger(__name__)

I2C_EEPROM_BUS = [0, 1, 2, 7]
MODULE_I2CBUS_TABLE = {
    'p3767-0005': 7, #'NVIDIA Jetson Orin Nano (Developer kit)',
    'p3767-0004': 7, #'NVIDIA Jetson Orin Nano (4GB ram)',
    'p3767-0003': 7, #'NVIDIA Jetson Orin Nano (8GB ram)',
    'p3767-0001': 7, #'NVIDIA Jetson Orin NX (8GB ram)',
    'p3767-0000': 7, #'NVIDIA Jetson Orin NX (16GB ram)',
    'p3701-0005': 7, #'NVIDIA Jetson AGX Orin (64GB ram)',
    'p3701-0004': 7, #'NVIDIA Jetson AGX Orin (32GB ram)',
    'p3701-0002': 7, #'NVIDIA Jetson IGX Orin (Developer kit)',
    'p3701-0000': 7, #'NVIDIA Jetson AGX Orin',
    'p3668-0003': 8, #'NVIDIA Jetson Xavier NX (16GB ram)',
    'p3668-0001': 8, #'NVIDIA Jetson Xavier NX',
    'p3668-0000': 8, #'NVIDIA Jetson Xavier NX (Developer kit)',
    'p2888-0008': 8, #'NVIDIA Jetson AGX Xavier Industrial (32 GB ram)',
    'p2888-0006': 8, #'NVIDIA Jetson AGX Xavier (8 GB ram)',
    'p2888-0005': 8, #'NVIDIA Jetson AGX Xavier (64 GB ram)',
    'p2888-0004': 8, #'NVIDIA Jetson AGX Xavier (32 GB ram)',
    'p2888-0003': 8, #'NVIDIA Jetson AGX Xavier (32 GB ram)',
    'p2888-0001': 8, #'NVIDIA Jetson AGX Xavier (16 GB ram)',
    'p3448-0003': 1, #'NVIDIA Jetson Nano (2 GB ram)',
    'p3448-0002': 1, #'NVIDIA Jetson Nano module (16Gb eMMC)',
    'p3448-0000': 1, #'NVIDIA Jetson Nano (4 GB ram)',
    'p3636-0001': 1, #'NVIDIA Jetson TX2 NX',
    'p3509-0000': 1, #'NVIDIA Jetson TX2 NX',
    'p3489-0888': 1, #'NVIDIA Jetson TX2 (4 GB ram)',
    'p3489-0000': 1, #'NVIDIA Jetson TX2i',
    'p3310-1000': 1, #'NVIDIA Jetson TX2',
    'p2180-1000': 0, #'NVIDIA Jetson TX1',
    'r375-0001': 1, #'NVIDIA Jetson TK1', https://jetsonhacks.com/2015/10/25/4-character-7-segment-led-over-i2c-nvidia-jetson-tk1/
    'p3904-0000': 99, #'NVIDIA Clara AGX',
    # Other modules
    'p2595-0000-A0': 99 #'Nintendo Switch'
}

def get_part_number():
    part_number = ''
    jetson_part_number = ''
    # Find 699-level part number from EEPROM and extract P-number
    for bus_number in I2C_EEPROM_BUS:
        try:
            bus = SMBus(bus_number)
            part_number = bus.read_i2c_block_data(0x50, 20, 29)
            part_number = ''.join(chr(i) for i in part_number).rstrip('\x00')
            # print(part_number)
            board_id = part_number[5:9]
            sku = part_number[10:14]
            jetson_part_number = "p{board_id}-{sku}".format(board_id=board_id, sku=sku)
            return part_number, jetson_part_number
        except (IOError, OSError):
            # print("Error I2C bus: {bus_number}".format(bus_number=bus_number))
            pass
    return part_number, jetson_part_number

class DisplayServer(object):
    
    def __init__(self, *args, **kwargs):

        part_number, jetson_part_number = get_part_number()
        i2c_bus_number = MODULE_I2CBUS_TABLE.get(jetson_part_number)
        logger.info(f"part_number: {part_number}, jetson_part_number: {jetson_part_number}")
        logger.info(f"i2c_bus_number = {i2c_bus_number}")
        if not i2c_bus_number:
            i2c_bus_number = 7  # Default: I2C bus 7 for Jetson AGX Orin

        self.display = Adafruit_SSD1306.SSD1306_128_32(rst=None, i2c_bus=i2c_bus_number, gpio=1)
        self.display.begin()
        self.display.clear()
        self.display.display()
        self.font = PIL.ImageFont.load_default()
        self.image = PIL.Image.new('1', (self.display.width, self.display.height))
        self.draw = PIL.ImageDraw.Draw(self.image)
        self.draw.rectangle((0, 0, self.image.width, self.image.height), outline=0, fill=0)
        self.stats_enabled = False
        self.stats_thread = None
        self.stats_interval = 1.0
        self.enable_stats()
        
    def _run_display_stats(self):
        while self.stats_enabled:
            self.draw.rectangle((0, 0, self.image.width, self.image.height), outline=0, fill=0)

            # set IP address
            top = -2
            if ip_address('eth0') is not None:
                self.draw.text((4, top), 'IP: ' + str(ip_address('eth0')), font=self.font, fill=255)
            elif ip_address('wlan0') is not None:
                self.draw.text((4, top), 'IP: ' + str(ip_address('wlan0')), font=self.font, fill=255)
            else:
                self.draw.text((4, top), 'IP: not available')

            top = 6
            power_mode_str = power_mode()
            self.draw.text((4, top), 'MODE: ' + power_mode_str, font=self.font, fill=255)
            
            # set stats headers
            top = 14
            offset = 3 * 8
            headers = ['PWR', 'CPU', 'GPU', 'RAM', 'DSK']
            for i, header in enumerate(headers):
                self.draw.text((i * offset + 4, top), header, font=self.font, fill=255)

            # set stats fields
            top = 22
            power_watts = '%.1f' % power_usage()
            gpu_percent = '%02d%%' % int(round(gpu_usage() * 100.0, 1))
            cpu_percent = '%02d%%' % int(round(cpu_usage() * 100.0, 1))
            ram_percent = '%02d%%' % int(round(memory_usage() * 100.0, 1))
            disk_percent = '%02d%%' % int(round(disk_usage() * 100.0, 1))
            
            entries = [power_watts, cpu_percent, gpu_percent, ram_percent, disk_percent]
            for i, entry in enumerate(entries):
                self.draw.text((i * offset + 4, top), entry, font=self.font, fill=255)

            self.display.image(self.image)
            self.display.display()

            time.sleep(self.stats_interval)
            
    def enable_stats(self):
        # start stats display thread
        if not self.stats_enabled:
            self.stats_enabled = True
            self.stats_thread = threading.Thread(target=self._run_display_stats)
            self.stats_thread.start()
        
    def disable_stats(self):
        self.stats_enabled = False
        if self.stats_thread is not None:
            self.stats_thread.join()
        self.draw.rectangle((0, 0, self.image.width, self.image.height), outline=0, fill=0)
        self.display.image(self.image)
        self.display.display()

    def set_text(self, text):
        self.disable_stats()
        self.draw.rectangle((0, 0, self.image.width, self.image.height), outline=0, fill=0)
        
        lines = text.split('\n')
        top = 2
        for line in lines:
            self.draw.text((4, top), line, font=self.font, fill=255)
            top += 10
        
        self.display.image(self.image)
        self.display.display()
        

server = DisplayServer()
app = Flask(__name__)


@app.route('/stats/on')
def enable_stats():
    global server
    server.enable_stats()
    return "stats enabled"

    
@app.route('/stats/off')
def disable_stats():
    global server
    server.disable_stats()
    return "stats disabled"


@app.route('/text/<text>')
def set_text(text):
    global server
    server.set_text(text)
    return 'set text: \n\n%s' % text


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5005', debug=False)

