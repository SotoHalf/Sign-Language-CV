import os
import sys
import uuid
import time
import socket
import hashlib
from dotenv import load_dotenv
from pathlib import Path


def generate_unique_id() -> str:
    """
    Generate a unique identifier combining machine fingerprint, nanosecond
    timestamp and random hex.

    :return: String in the format ``<machine_hash>_<timestamp_ns>_<random>``.
    :rtype: str
    """
    hostname: str = socket.gethostname()
    machine_hash: str = hashlib.md5(hostname.encode()).hexdigest()[:6]
    timestamp: int = time.time_ns()
    random_part: str = uuid.uuid4().hex[:6]
    return f"{machine_hash}_{timestamp}_{random_part}"


class AppPaths:
    """
    Centralizes project-root path resolution.

    Works both in normal Python execution and when the project is packaged
    with PyInstaller (``sys.frozen`` is set in that case).
    """

    # When frozen by PyInstaller the executable sits at the project root;
    # in normal execution we resolve two levels up from this file.
    if getattr(sys, "frozen", False):
        ROOT: Path = Path(sys.executable).parent
    else:
        ROOT: Path = Path(__file__).resolve().parents[1]

    @classmethod
    def path(cls, *parts: str) -> str:
        """
        Return an absolute path by joining the project root with the given parts.

        :param parts: Path components to join (e.g. ``"data"``, ``"processed"``).
        :type parts: str
        :return: Absolute path string.
        :rtype: str
        """
        return str(cls.ROOT.joinpath(*parts))

    @classmethod
    def load_env(cls) -> None:
        """
        Load environment variables from the ``.env`` file at the project root.
        Does nothing if the file does not exist.
        """
        env: str = cls.path(".env")
        if Path(env).exists():
            load_dotenv(env)
