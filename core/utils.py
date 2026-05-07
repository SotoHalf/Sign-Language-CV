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
    Generate a unique ID based on machine
    """

    hostname = socket.gethostname()
    machine_hash = hashlib.md5(hostname.encode()).hexdigest()[:6]
    timestamp = time.time_ns()
    random_part = uuid.uuid4().hex[:6]

    return f"{machine_hash}_{timestamp}_{random_part}" 

class AppPaths:
    if getattr(sys, "frozen", False):
        ROOT = Path(sys.executable).parent
        #ROOT = Path(sys._MEIPASS)
    else:
        ROOT = Path(__file__).resolve().parents[1]

    @classmethod
    def path(cls, *parts):
        return str(cls.ROOT.joinpath(*parts))

    @classmethod
    def load_env(cls):
        env = cls.path(".env")
        if Path(env).exists():
            load_dotenv(env)

"""
def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    return project_root

def get_project_root() -> None:
    '''
    Returns the absolute path to the project's root
    Works both in normal execution and when packaged with PyInstaller
    '''
    if getattr(sys, 'frozen', False):
        # running as a PyInstaller, use the temp folder
        # NOT TESTED
        base_path = sys._MEIPASS
    else:
        # normal execution, use the directory of this file
        base_path = os.path.dirname(os.path.abspath(__file__))

    if not base_path:
        return None
    
    return os.path.abspath(os.path.join(base_path, ".."))

def load_env(dotenv_filename: str = ".env") -> None:
    '''
    Load environment variables from a .env file.
    Works both in normal execution and in PyInstaller.

    :param dotenv_filename: Name of the .env file (default: ".env")
    '''

    project_root = get_project_root()
    if project_root:
        dotenv_path = os.path.join(project_root, dotenv_filename)

        # Load the .env if it exists
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)

"""