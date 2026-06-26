"""ROS message timestamp helpers."""


def copy_stamp(dst_stamp, src_stamp) -> None:
    """Copy sec/nanosec fields to avoid cross-package Time assignment issues."""
    dst_stamp.sec = int(src_stamp.sec)
    dst_stamp.nanosec = int(src_stamp.nanosec)
