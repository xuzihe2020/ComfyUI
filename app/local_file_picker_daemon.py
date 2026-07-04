import argparse
import ctypes
import json
import os
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer


IMAGE_FILETYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff"),
    ("All files", "*.*"),
]


def _parent_is_alive(parent_pid):
    if parent_pid <= 0:
        return True

    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            parent_pid,
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            still_active = 259
            return exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    try:
        os.kill(parent_pid, 0)
    except OSError:
        return False
    return True


def _monitor_parent(parent_pid, server):
    while _parent_is_alive(parent_pid):
        threading.Event().wait(1.0)
    os._exit(0)


def _open_file_picker(initial_dir):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass

    options = {
        "title": "Select image",
        "filetypes": IMAGE_FILETYPES,
    }
    if initial_dir and os.path.isdir(initial_dir):
        options["initialdir"] = initial_dir

    try:
        return filedialog.askopenfilename(**options)
    finally:
        root.destroy()


class LocalFilePickerHandler(BaseHTTPRequestHandler):
    server_version = "ComfyUILocalFilePicker/1.0"

    def log_message(self, format, *args):
        return

    def _send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/health":
            self._send_json(200, {"ok": True, "pid": os.getpid()})
            return

        if parsed.path == "/shutdown":
            self._send_json(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        if parsed.path == "/pick-file":
            initial_dir = query.get("initial_dir", [""])[0]
            try:
                path = _open_file_picker(initial_dir)
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
                return

            self._send_json(200, {"ok": True, "path": path or ""})
            return

        self._send_json(404, {"ok": False, "error": "not found"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), LocalFilePickerHandler)
    threading.Thread(target=_monitor_parent, args=(args.parent_pid, server), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
