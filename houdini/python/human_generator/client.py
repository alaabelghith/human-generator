"""Talks to the local Human Generator server (server/hg_server.py). Houdini never imports torch."""
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np
import hou

# a system proxy can answer 403 for 127.0.0.1: always connect directly
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def repo_root():
    root = os.environ.get("HUMAN_GENERATOR") or hou.getenv("HUMAN_GENERATOR")
    if not root:
        raise hou.NodeError("HUMAN_GENERATOR is not set. Run install.ps1 (it installs the Houdini package), then restart Houdini.")
    return root


def _server_script():
    root = repo_root()
    return os.path.join(root, "server", "start_server.bat" if os.name == "nt" else "start_server.sh")


def is_up(url):
    try:
        return _OPENER.open(url + "/ping", timeout=2).status == 200
    except Exception:
        return False


def ensure_server(url, wait=180):
    """Start the server in its own (minimised) console if it does not answer /ping, then wait for it."""
    if is_up(url):
        return
    script = _server_script()
    if not os.path.exists(script):
        raise hou.NodeError("Server script not found: %s" % script)
    # Houdini's own Python settings must not leak into the server's virtual environment
    env = {k: v for k, v in os.environ.items()
           if not (k.upper() in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP")
                   or k.upper().startswith(("HOUDINI", "HFS", "HH")))}
    if os.name == "nt":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 7  # minimised
        subprocess.Popen(["cmd.exe", "/c", script], creationflags=subprocess.CREATE_NEW_CONSOLE,
                         startupinfo=si, cwd=os.path.dirname(script), env=env)
    else:
        subprocess.Popen(["bash", script], cwd=os.path.dirname(script), env=env, start_new_session=True)
    with hou.InterruptableOperation("Starting the Human Generator server (first start can take a minute)",
                                    open_interrupt_dialog=True):
        t0 = time.time()
        while time.time() - t0 < wait:
            if is_up(url):
                return
            time.sleep(1.5)
    raise hou.NodeError("The server did not start within %d s. Run %s by hand to see the error." % (wait, script))


def post(url, route, req, timeout=600):
    data = json.dumps(req).encode()
    try:
        r = _OPENER.open(urllib.request.Request(url + route, data, {"Content-Type": "application/json"}), timeout=timeout)
    except urllib.error.HTTPError as e:
        raise hou.NodeError("Server error: %s" % e.read().decode(errors="replace")[:600])
    except urllib.error.URLError as e:
        raise hou.NodeError("Server not reachable at %s (%s)." % (url, e))
    return json.loads(r.read())


def load_static(path):
    """Topology, UVs, skin weights and skeleton for one LOD, converted to Houdini winding. Cached in hou.session."""
    c = getattr(hou.session, "_hg_static", {})
    if path not in c:
        z = np.load(path)
        d = {k: z[k] for k in z.files}
        counts = d["face_counts"]
        polys = np.split(d["face_indices"], np.cumsum(counts)[:-1])
        d["_polys"] = [p[::-1].tolist() for p in polys]  # Houdini winding
        if "corner_uv" in d:
            uv = np.concatenate([d["corner_uv"], np.zeros((len(d["corner_uv"]), 1), np.float32)], 1)
            off = np.concatenate([[0], np.cumsum(counts)[:-1]])
            order = np.concatenate([np.arange(o + n - 1, o - 1, -1) for o, n in zip(off, counts)])
            d["_uv"] = uv[order].astype(np.float32).tobytes()
        c[path] = d
        hou.session._hg_static = c
    return c[path]
