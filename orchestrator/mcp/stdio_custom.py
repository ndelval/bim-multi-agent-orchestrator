"""
Custom stdio transport for MCP that works with .NET servers.

The official MCP SDK's stdio_client has issues with some .NET MCP servers.
This implementation uses direct subprocess management.
"""
import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class StdioReadStream:
    """Async stream reader for MCP messages from stdout."""

    def __init__(self, stdout: asyncio.StreamReader):
        self._stdout = stdout
        self._closed = False

    async def receive(self):
        """Receive next message from stdout."""
        if self._closed:
            raise RuntimeError("Stream is closed")

        line = await self._stdout.readline()
        if not line:
            raise EOFError("Stream ended")

        return line

    def close(self):
        """Mark stream as closed."""
        self._closed = True


class StdioWriteStream:
    """Async stream writer for MCP messages to stdin."""

    def __init__(self, stdin: asyncio.StreamWriter):
        self._stdin = stdin
        self._closed = False

    async def send(self, data: bytes):
        """Send message to stdin."""
        if self._closed:
            raise RuntimeError("Stream is closed")

        self._stdin.write(data)
        await self._stdin.drain()

    def close(self):
        """Close the stdin stream."""
        if not self._closed:
            self._stdin.close()
            self._closed = True


@asynccontextmanager
async def custom_stdio_client(
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None
):
    """
    Create a custom stdio MCP client that works with .NET servers.

    Args:
        command: Command to execute (e.g., "dotnet")
        args: Command arguments
        env: Environment variables

    Yields:
        Tuple of (read_stream, write_stream)
    """
    args = args or []
    full_env = None
    if env:
        import os
        full_env = os.environ.copy()
        full_env.update(env)

    logger.debug(f"Starting subprocess: {command} {' '.join(args)}")

    # Start subprocess with pipes
    process = await asyncio.create_subprocess_exec(
        command,
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=full_env
    )

    logger.debug(f"Subprocess started with PID {process.pid}")

    # Create stream wrappers
    read_stream = StdioReadStream(process.stdout)
    write_stream = StdioWriteStream(process.stdin)

    try:
        # Give the server a moment to initialize
        await asyncio.sleep(0.5)

        yield (read_stream, write_stream)

    finally:
        # Cleanup
        logger.debug(f"Cleaning up subprocess {process.pid}")

        write_stream.close()
        read_stream.close()

        # Terminate process
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"Process {process.pid} did not terminate, killing...")
            process.kill()
            await process.wait()

        logger.debug(f"Subprocess {process.pid} cleaned up")
