"""Generic WebSocket connection management for the Stockpile API."""

from fastapi import WebSocket


class WebSocketManager:
    """Manages WebSocket connections grouped by key (e.g., job_id, search_id).

    This class deduplicates the repeated connect/broadcast/disconnect pattern
    used across multiple routers for real-time updates.
    """

    def __init__(self):
        """Initialize the WebSocket manager with empty connections dict."""
        self.connections: dict[str, list[WebSocket]] = {}

    async def connect(self, key: str, websocket: WebSocket) -> None:
        """Accept a WebSocket connection and add it to the connection pool.

        Args:
            key: The grouping key (e.g., job_id, search_id)
            websocket: The WebSocket connection to add
        """
        await websocket.accept()
        if key not in self.connections:
            self.connections[key] = []
        self.connections[key].append(websocket)

    async def broadcast(self, key: str, message: dict) -> None:
        """Broadcast a message to all connected WebSockets for a given key.

        Args:
            key: The grouping key
            message: The message dict to send as JSON

        Note:
            Automatically cleans up disconnected WebSockets.
        """
        disconnected = []
        for ws in self.connections.get(key, []):
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        # Clean up disconnected WebSockets
        for ws in disconnected:
            if ws in self.connections.get(key, []):
                self.connections[key].remove(ws)

    def disconnect(self, key: str, websocket: WebSocket) -> None:
        """Remove a WebSocket from the connection pool.

        Args:
            key: The grouping key
            websocket: The WebSocket connection to remove
        """
        if key in self.connections and websocket in self.connections[key]:
            self.connections[key].remove(websocket)

    def cleanup(self, key: str) -> None:
        """Remove all connections for a given key.

        Args:
            key: The grouping key to clean up
        """
        self.connections.pop(key, None)

    def has_key(self, key: str) -> bool:
        """Check if a key exists in the connection pool.

        Args:
            key: The grouping key to check

        Returns:
            True if the key exists, False otherwise
        """
        return key in self.connections

    def ensure_key(self, key: str) -> None:
        """Ensure a key exists in the connection pool.

        Args:
            key: The grouping key to ensure exists
        """
        if key not in self.connections:
            self.connections[key] = []
