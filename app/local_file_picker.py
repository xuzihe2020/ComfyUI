import atexit
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


HOST = "127.0.0.1"
PORT = int(os.environ.get("COMFYUI_LOCAL_FILE_PICKER_PORT", "31987"))
BASE_URL = f"http://{HOST}:{PORT}"

_started_process = None
_shutdown_in_progress = False


def _request_json(path, timeout=0.5):
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def is_daemon_running():
    try:
        data = _request_json("/health", timeout=0.25)
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False
    return bool(data.get("ok"))


def ensure_daemon():
    global _started_process

    if is_daemon_running():
        return True

    daemon_script = os.path.join(os.path.dirname(__file__), "local_file_picker_daemon.py")
    cmd = [
        sys.executable,
        daemon_script,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--parent-pid",
        str(os.getpid()),
    ]

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    _started_process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    for _ in range(30):
        if is_daemon_running():
            logging.info("Local file picker daemon started on %s", BASE_URL)
            return True
        if _started_process.poll() is not None:
            break
        time.sleep(0.1)

    logging.warning("Local file picker daemon did not start")
    return False


def _wait_for_process(process, timeout):
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            process.wait(timeout=min(remaining, 0.1))
        except subprocess.TimeoutExpired:
            pass
        except KeyboardInterrupt:
            logging.info("Interrupted while stopping local file picker daemon; forcing shutdown")
            return False
    return True


def _force_stop_process(process):
    if process.poll() is not None:
        return

    try:
        process.terminate()
    except OSError:
        pass

    if _wait_for_process(process, 1):
        return

    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass

    _wait_for_process(process, 1)


def shutdown_daemon():
    global _started_process, _shutdown_in_progress
    if _shutdown_in_progress or _started_process is None:
        return

    _shutdown_in_progress = True
    process = _started_process
    _started_process = None

    try:
        _request_json("/shutdown", timeout=0.25)
    except KeyboardInterrupt:
        logging.info("Interrupted while requesting local file picker daemon shutdown; forcing shutdown")
    except Exception:
        pass

    try:
        if not _wait_for_process(process, 2):
            _force_stop_process(process)
    finally:
        _shutdown_in_progress = False


def pick_file(initial_dir=""):
    ensure_daemon()
    params = urllib.parse.urlencode({"initial_dir": initial_dir or ""})
    data = _request_json(f"/pick-file?{params}", timeout=None)
    if not data.get("ok"):
        raise RuntimeError(data.get("error") or "Local file picker failed")
    return data.get("path") or ""


atexit.register(shutdown_daemon)
