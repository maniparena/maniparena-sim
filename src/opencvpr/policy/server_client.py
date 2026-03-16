"""WebSocket inference client for remote policy servers."""

from __future__ import annotations

from typing import Any, Dict


class PolicyServerClient:
    """Sync WebSocket client using msgpack serialization.

    Usage::

        client = PolicyServerClient('localhost', 8000)
        client.connect()
        response = client.predict(observation_dict)
        client.close()
    """

    def __init__(
        self,
        address: str = 'localhost',
        port: int = 8000,
    ):
        """Create client for the given address and port."""
        self.address = address
        self.port = port
        self.uri = f'ws://{address}:{port}'
        self._connection = None
        self._metadata: Dict[str, Any] = {}
        self._connected = False

    @property
    def connected(self) -> bool:
        """Whether the client is connected."""
        return self._connected

    @property
    def metadata(self) -> Dict[str, Any]:
        """Server metadata received on connect."""
        return self._metadata

    def connect(self) -> Dict[str, Any]:
        """Connect to server and return metadata."""
        import msgpack
        import msgpack_numpy
        from websockets.sync.client import connect as ws_connect

        msgpack_numpy.patch()
        print(
            f'[PolicyServerClient] Connecting to {self.uri}'
        )
        self._connection = ws_connect(
            self.uri, max_size=None,
        )
        metadata_bytes = self._connection.recv()
        self._metadata = msgpack.unpackb(metadata_bytes)
        self._connected = True
        print(
            f'[PolicyServerClient] Connected to {self.uri}'
        )
        return self._metadata

    def ensure_connected(self) -> None:
        """Connect if not already connected."""
        if not self._connected:
            self.connect()

    def predict(
        self, observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send observation, receive action prediction."""
        import msgpack

        if self._connection is None:
            raise RuntimeError(
                'Not connected. Call connect() first.'
            )
        self._connection.send(msgpack.packb(observation))
        response_bytes = self._connection.recv()
        if isinstance(response_bytes, str):
            raise RuntimeError(
                f'Server error: {response_bytes}'
            )
        return msgpack.unpackb(response_bytes)

    def close(self) -> None:
        """Close the connection."""
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
        self._connection = None
        self._connected = False

    def __del__(self):
        """Clean up on garbage collection."""
        self.close()
