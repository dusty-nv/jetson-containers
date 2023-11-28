
## Usage

### Pre-setup

1. Make sure you have OLED display module connected onto the 40pin at the right location (over Pin `1` - `6`).
2. `sudo usermod -aG i2c $USER` (really needed?)

### How to start

```
./run.sh $(./autotag oled)
```

## Options: Running on Startup



## Troubleshooting

### `OSError: [Errno 121] Remote I/O error`

```
Traceback (most recent call last):
  File "/opt/oled/oled_display_server.py", line 97, in <module>
    server = DisplayServer()
  File "/opt/oled/oled_display_server.py", line 15, in __init__
    self.display.begin()
  File "/usr/local/lib/python3.8/dist-packages/Adafruit_SSD1306/SSD1306.py", line 148, in begin
    self._initialize()
  File "/usr/local/lib/python3.8/dist-packages/Adafruit_SSD1306/SSD1306.py", line 292, in _initialize
    self.command(SSD1306_DISPLAYOFF)                    # 0xAE
  File "/usr/local/lib/python3.8/dist-packages/Adafruit_SSD1306/SSD1306.py", line 129, in command
    self._i2c.write8(control, c)
  File "/usr/local/lib/python3.8/dist-packages/Adafruit_GPIO/I2C.py", line 114, in write8
    self._bus.write_byte_data(self._address, register, value)
  File "/usr/local/lib/python3.8/dist-packages/Adafruit_PureIO/smbus.py", line 316, in write_byte_data
    self._device.write(data)
OSError: [Errno 121] Remote I/O error
```

The OLED module may not be inserted to the 40-pin at the correct location.<br>
Check if the OLED module is connected and shows up on the I2C bus by:

```
i2cdetect -y -r 7
```

Use `7` for Orin generation developer kits, `8` for Xavier generation developer kits.

You should see find a device at address `0x3c` like the following.

```
$ i2cdetect -y -r 7
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:          -- -- -- -- -- -- -- -- -- -- -- -- -- 
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
30: -- -- -- -- -- -- -- -- -- -- -- -- 3c -- -- -- 
40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
70: -- -- -- -- -- -- -- --
```





