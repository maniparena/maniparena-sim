"""Lightweight debug print shim."""


def maniparenaprint(*args, **kwargs):
    """Print debug messages to stdout."""
    print(*args, **kwargs)
