from pkg_resources import get_distribution

print("testing homeassistant...")

import homeassistant

print(f"Home Assistant version: {get_distribution("homeassistant").version}")

print("homeassistant OK")
