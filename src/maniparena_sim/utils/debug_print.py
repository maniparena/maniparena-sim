"""Lightweight debug print shim."""


def manaprint(*args, **kwargs):
    """Print debug messages to stdout."""
    print(*args, **kwargs)
