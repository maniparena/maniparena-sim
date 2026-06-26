"""Constants used by the ROS bridge and related utilities.

Isaac Lab outputs RGB as 0-1 float and depth as meters (float32).
ROS expects RGB as 0-255 uint8 and depth as millimeters (uint16).
"""

# Floating point epsilon for numerical comparisons
FLOAT_EPS = 1e-6

# RGB image constants
RGB_MAX_VALUE = 255
RGB_DTYPE = "uint8"
RGB_CHANNELS = 3

# Standard gravity (m/s²) — used to convert linear acceleration from m/s² to g
STANDARD_GRAVITY = 9.80665

# Depth image constants
DEPTH_DTYPE = "uint16"
DEPTH_MAX_MM = 65535  # uint16 max value
