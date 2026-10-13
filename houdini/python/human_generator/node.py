"""The Human Generator node: body in real units + Kimodo motion, cooked by a Python-type SOP.

cook(node)      PythonCook entry point. Output 0 body, 1 skeleton, 2 both.
generate(node)  "Generate Motion" button: prompt (+ optional input curve) -> Kimodo -> motion file on the node.

The node sends its sliders to the local server and reads back NPZ files. Results are cached per
(body settings, motion), so scrubbing the timeline never calls the server.
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import hou

from . import client

J_LEYE, J_REYE = 10, 11  # SOMA-X joint indices
ANNY_PHENO = ("gender", "age", "muscle", "weight", "height", "proportions", "cupsize", "firmness")
LODS = ("mid", "low", "xlo")
N_PC = 12
SOMA_SCALE_GROUPS = {  # SOMA backend sliders -> SOMA scale names
    "sc_upperarm": ("LeftArm", "RightArm"),
    "sc_forearm": ("LeftForeArm", "RightForeArm"),
    "sc_hand": ("LeftHand", "RightHand"),
    "sc_shin": ("LeftShin", "RightShin"),
    "sc_foot": ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"),
}
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


# ------------------------------------------------------------------------------------------------ helpers
def _ev(node, name, default=None):
    p = node.parm(name)
    return p.eval() if p is not None else default


def parse_local(text):
    """'torso-vshape-incr=0.5, hip-scale-horiz-incr=-0.3' -> {name: value} (Anny local changes)."""
    out = {}
    for part in (text or "").replace(";", ",").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def mat_to_aa(R):
    """Rotation matrices (..., 3, 3) -> axis-angle (..., 3), robust near 180 degrees."""
    R = np.asarray(R, np.float64)
    th = np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(th)[..., None]
    small = np.abs(s) < 1e-6
    aa = np.where(small, v / 2, v / np.where(small, 1, 2 * s) * th[..., None])
    for i in zip(*np.nonzero(th > np.pi - 1e-3)):
        M = R[i]
        w = np.sqrt(np.clip((np.diag(M) + 1) / 2, 0, None))
        k = int(np.argmax(w))
        a = M[:, k] + M[k, :]
        a[k] = 2 * w[k] ** 2
        aa[i] = a / np.linalg.norm(a) * th[i]
    return aa.astype(np.float32)


def _body_request(node):
    lod = LODS[node.evalParm("lod")]
    if node.evalParm("body_model") == 1:  # Anny: real units
        req = {"lod": lod, "backend": "anny", "correctives": True, "global_scale": node.evalParm("height"),
               "phenotype": {k: node.evalParm("ph_" + k) for k in ANNY_PHENO if node.parm("ph_" + k) is not None},
               "local_changes": parse_local(node.evalParm("anny_local")),
               "units": {"age_years": node.evalParm("age_years"), "height_cm": node.evalParm("height_cm"),
                         "weight_kg": node.evalParm("weight_kg")}}
        b = node.evalParm("buttocks")
        if abs(b) > 1e-6:
            req["local_changes"]["buttocks-volume-incr"] = req["local_changes"].get("buttocks-volume-incr", 0.0) + b
        return req
    ident = np.zeros(128, np.float32)  # SOMA PCA identity
    if node.evalParm("rand_amount") > 0:
        ident += np.random.default_rng(int(node.evalParm("rand_seed"))).standard_normal(128).astype(np.float32) * node.evalParm("rand_amount")
    for i in range(N_PC):
        ident[i] += node.evalParm("pc_%d" % i)
    return {"lod": lod, "backend": "soma", "identity": ident.tolist(), "global_scale": node.evalParm("height"), "correctives": True}


def _soma_scales(node, names):
    s = np.ones(len(names), np.float32)
    for parm, targets in SOMA_SCALE_GROUPS.items():
        for i, n in enumerate(names):
            if n in targets:
                s[i] = node.evalParm(parm)
    for i, n in enumerate(names):
        if any(n.startswith(side + "Hand" + f) for side in ("Left", "Right") for f in FINGERS):
            s[i] = node.evalParm("sc_fingers")
    return s.tolist()


def _hip_height(rest_npz):
    rest = np.load(rest_npz)
    return float(rest["rest_xforms"][1, 1, 3] - rest["rest_verts"][:, 1].min())


# ------------------------------------------------------------------------------------------------ posing
def posed(node):
    """Vertices and joint transforms for every frame of the current motion (or the rest pose). Cached."""
    url = node.evalParm("server").rstrip("/")
    req = _body_request(node)
    anny = req["backend"] == "anny"
    mpath = node.evalParm("motion").strip()
    use_motion = bool(node.evalParm("use_motion")) and bool(mpath)
    if use_motion and not os.path.exists(mpath):
        raise hou.NodeError("Motion file not found: %s\nPress Generate Motion, or untick Use Motion." % mpath)
    key = json.dumps([req, use_motion, mpath, os.path.getmtime(mpath) if use_motion else 0,
                      node.evalParm("inplace"), node.evalParm("scale_stride"), node.evalParm("ground"),
                      [] if anny else [node.evalParm(p) for p in SOMA_SCALE_GROUPS] + [node.evalParm("sc_fingers")]],
                     sort_keys=True)
    cache = getattr(hou.session, "_hg_pose", {})
    if key in cache:
        return cache[key]

    client.ensure_server(url)
    first = client.post(url, "/body", req)               # rest body; also creates the topology file
    st = client.load_static(first["static"])
    if not anny:
        req["scales"] = _soma_scales(node, [str(n) for n in st["scale_names"]])
    if use_motion:
        z = np.load(mpath)
        req["poses"] = mat_to_aa(z["local_rot_mats"]).tolist()
        tr = z["root_positions"].astype(np.float32).copy()
        hip_body, hip_kim = _hip_height(first["npz"]), float(np.median(z["root_positions"][:, 1]))
        ratio = hip_body / hip_kim if hip_kim > 0.2 and hip_body > 0.1 else 1.0
        side = os.path.splitext(mpath)[0] + ".path.json"
        on_path = os.path.exists(side)
        if on_path:  # Kimodo walked the input curve, relative to its start: put it back on the curve
            ox, oz = json.load(open(side))["origin"]
            tr[:, 1] *= ratio
            tr[:, 0] += ox
            tr[:, 2] += oz
        elif node.evalParm("scale_stride"):  # Kimodo moves a ~0.95 m-hip adult: scale the root path to this body
            tr *= ratio
        if node.evalParm("inplace") and not on_path:
            tr[:, 0] = tr[0, 0]
            tr[:, 2] = tr[0, 2]
        req["transl"] = tr.tolist()
    with hou.InterruptableOperation("Human Generator: posing the body", open_interrupt_dialog=True):
        res = client.post(url, "/body", req)
    o = np.load(res["npz"])
    verts = o["verts"] if use_motion else o["rest_verts"][None]
    X = (o["xforms"] if use_motion else o["rest_xforms"][None]).copy()
    rest_X = o["rest_xforms"].copy()
    if node.evalParm("ground"):
        dy = -float(verts[:, :, 1].min())
        verts = verts + np.float32([0, dy, 0])
        X[:, :, 1, 3] += dy
        rest_X[:, 1, 3] -= float(o["rest_verts"][:, 1].min())
    d = {"verts": verts.astype(np.float32), "X": X, "rest_X": rest_X, "st": st, "measured": res.get("anny") or {}}
    hou.session._hg_pose = {key: d}  # one pose set in memory
    return d


def _frame(node, n):
    f = int(round(hou.frame())) - node.evalParm("start_frame")
    return f % n if node.evalParm("loop") else int(np.clip(f, 0, n - 1))


# ------------------------------------------------------------------------------------------------ cook
def cook(node):
    geo = node.geometry()
    geo.clear()  # the optional path curve on input 0 is copied into the output by default
    mode = node.evalParm("output")
    d = posed(node)
    f = _frame(node, d["verts"].shape[0])
    if mode in (0, 2):
        _cook_body(node, geo, d, f)
    if mode in (1, 2):
        _cook_skeleton(geo, d, f)


def _cook_body(node, geo, d, f):
    st = d["st"]
    v = d["verts"][f]
    if node.evalParm("smooth_iter") > 0:
        v = _relax(v, st, node.evalParm("smooth_iter"), node.evalParm("smooth_strength"))
    geo.createPoints([(0.0, 0.0, 0.0)] * len(v))
    geo.setPointFloatAttribValuesFromString("P", v.astype(np.float32).tobytes())
    geo.createPolygons(st["_polys"])
    if "_uv" in st:
        geo.addAttrib(hou.attribType.Vertex, "uv", (0.0, 0.0, 0.0))
        geo.setVertexFloatAttribValuesFromString("uv", st["_uv"])
    geo.createPrimGroup("body").add(geo.prims())
    for k in ("height_cm", "weight_kg", "age_years", "scale"):
        if d["measured"].get(k) is not None:
            geo.addAttrib(hou.attribType.Global, k, 0.0)
            geo.setGlobalAttribValue(k, float(d["measured"][k]))
    eye_ranges = _add_eyes(geo, d, f) if node.evalParm("eyes") else []
    if node.evalParm("smooth_normals"):
        _smooth_normals(geo, v, st, eye_ranges)


def _cook_skeleton(geo, d, f):
    X, st = d["X"][f], d["st"]
    pts = geo.createPoints([tuple(map(float, x[:3, 3])) for x in X])
    geo.addAttrib(hou.attribType.Point, "name", "")
    geo.addAttrib(hou.attribType.Point, "transform", (1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0))
    for p, n, x in zip(pts, st["joint_names"], X):
        p.setAttribValue("name", str(n))
        p.setAttribValue("transform", tuple(float(a) for a in x[:3, :3].T.reshape(-1)))  # row-vector convention
    for j, pj in enumerate(st["joint_parents"]):
        if pj != j:
            poly = geo.createPolygon(False)
            poly.addVertex(pts[int(pj)])
            poly.addVertex(pts[j])


# ------------------------------------------------------------------------------------------------ eyes
def _sphere(r, nu=32, nv=20):
    """UV sphere, pole on +Z (the gaze), outward-facing polygons."""
    pts = []
    for j in range(nv + 1):
        th = np.pi * j / nv
        for i in range(nu):
            ph = 2 * np.pi * i / nu
            pts.append((r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)))
    polys = []
    for j in range(nv):
        for i in range(nu):
            a, b = j * nu + i, j * nu + (i + 1) % nu
            c, e = (j + 1) * nu + (i + 1) % nu, (j + 1) * nu + i
            polys.append([a, b, c, e])
    return np.array(pts, np.float32), polys, nv, nu


def _add_eyes(geo, d, f):
    """Two eyeballs on the LeftEye / RightEye joints, sized from the eye spacing, carried by the joints."""
    restX, X = d["rest_X"], d["X"][f]
    spacing = float(np.linalg.norm(restX[J_LEYE, :3, 3] - restX[J_REYE, :3, 3]))
    base, polys, nv, nu = _sphere(0.0118 * (spacing / 0.0616))  # ~24 mm eyeball for ~62 mm eye spacing
    iris_rows = max(1, int(nv * 0.22))
    ranges = []
    for side, j in (("left_eye", J_LEYE), ("right_eye", J_REYE)):
        M = X[j] @ np.linalg.inv(restX[j])  # how the eye joint moved since rest
        c = restX[j, :3, 3]
        P = (base + c) @ M[:3, :3].T + M[:3, 3]
        start = len(geo.points())
        geo.createPoints([tuple(float(x) for x in p) for p in P])
        prims = geo.createPolygons([[start + i for i in q] for q in polys])
        for gname, members in ((side, prims), ("eyes", prims), ("iris", prims[:iris_rows * nu]),
                               ("pupil", prims[:max(1, iris_rows // 2) * nu])):
            (geo.findPrimGroup(gname) or geo.createPrimGroup(gname)).add(list(members))
        ranges.append((start, len(P), c @ M[:3, :3].T + M[:3, 3]))
    return ranges


# ------------------------------------------------------------------------------------------------ surface
def _smooth_normals(geo, v, st, eye_ranges):
    """Point normals: body = averaged face normals, eyes = radial."""
    if "_nidx" not in st:
        st["_nidx"] = np.array([(p[0], p[1], p[2], p[3] if len(p) > 3 else p[2]) for p in st["_polys"]], np.int64)
    idx = st["_nidx"]
    P = v.astype(np.float64)
    fn = np.cross(P[idx[:, 2]] - P[idx[:, 0]], P[idx[:, 3]] - P[idx[:, 1]])
    N = np.zeros_like(P)
    for k in range(4):
        np.add.at(N, idx[:, k], fn)
    radial = P - P.mean(0)
    radial[:, 1] = 0.0
    if np.sum(np.einsum("ij,ij->i", N, radial)) < 0:  # face away from the body's centre line
        N = -N
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    allN = np.zeros((len(geo.points()), 3), np.float32)
    allN[:len(P)] = N
    Pall = np.frombuffer(geo.pointFloatAttribValuesAsString("P"), np.float32).reshape(-1, 3)
    for start, count, centre in eye_ranges:
        dd = Pall[start:start + count] - np.asarray(centre, np.float32)
        allN[start:start + count] = dd / np.maximum(np.linalg.norm(dd, axis=1, keepdims=True), 1e-12)
    geo.addAttrib(hou.attribType.Point, "N", (0.0, 0.0, 0.0))
    geo.setPointFloatAttribValuesFromString("N", allN.astype(np.float32).tobytes())


def _relax(v, st, iters, strength):
    """Taubin smoothing (lambda/mu): relaxes folds without shrinking the body."""
    if "_edges" not in st:
        e = []
        for p in st["_polys"]:
            e += [(p[k], p[(k + 1) % len(p)]) for k in range(len(p))]
        e = np.unique(np.sort(np.array(e, np.int64), axis=1), axis=0)
        st["_edges"] = e
        st["_deg"] = np.maximum(np.bincount(e.ravel(), minlength=len(v)).astype(np.float64), 1.0)
    e, deg = st["_edges"], st["_deg"]
    P = v.astype(np.float64).copy()
    lam, mu = float(strength), -float(strength) - 0.02
    for _ in range(int(iters)):
        for w in (lam, mu):
            avg = np.zeros_like(P)
            np.add.at(avg, e[:, 0], P[e[:, 1]])
            np.add.at(avg, e[:, 1], P[e[:, 0]])
            P += w * (avg / deg[:, None] - P)
    return P.astype(np.float32)


# ------------------------------------------------------------------------------------------------ motion
def _path_from_input(node):
    """Optional input 0: a curve on the ground. Returns {xz (relative to its start), origin, shares} or None."""
    if not node.evalParm("follow_path"):
        return None
    ins = node.inputs()
    if not ins or ins[0] is None:
        return None
    prims = [p for p in ins[0].geometry().prims()
             if p.type() in (hou.primType.Polygon, hou.primType.NURBSCurve, hou.primType.BezierCurve)]
    if not prims:
        raise hou.NodeError("The path input has no curve.")
    pr = max(prims, key=lambda p: p.numVertices())
    if pr.type() == hou.primType.Polygon:
        P = np.array([v.point().position() for v in pr.vertices()], np.float64)
    else:
        P = np.array([pr.positionAt(u) for u in np.linspace(0, 1, 400)], np.float64)
    if node.evalParm("path_reverse"):
        P = P[::-1]
    o = P[0]
    shares = [float(x) for x in node.evalParm("path_share").replace(",", " ").split()] or None
    return {"xz": (P[:, [0, 2]] - o[[0, 2]]).tolist(), "origin": [float(o[0]), float(o[2])], "shares": shares}


def generate(node):
    """Generate Motion button: prompt -> Kimodo on the server -> motion file stored on the node."""
    prompt = node.evalParm("prompt").strip()
    if not prompt:
        raise hou.Error("Type a motion prompt first.")
    url = node.evalParm("server").rstrip("/")
    client.ensure_server(url)
    sentences = [x for x in prompt.split(".") if x.strip()]
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:40] or "motion"
    # every parameter is read HERE, on Houdini's main thread: hou must never be touched from the worker thread
    req = {"prompt": prompt, "duration": node.evalParm("duration"), "seed": node.evalParm("seed"),
           "transition_frames": node.evalParm("transition"), "inplace": True,
           "stem": "%s_s%d_%gs" % (slug, node.evalParm("seed"), node.evalParm("duration"))}
    durs = node.evalParm("durations").replace(",", " ").split()
    if durs:
        req["durations"] = [float(x) for x in durs]
        if len(req["durations"]) != len(sentences):
            raise hou.Error("%d sentences in the prompt but %d durations. Give one duration per sentence, e.g. '3 2 2'."
                            % (len(sentences), len(req["durations"])))
        req["stem"] = "%s_s%d_seq%d" % (slug, node.evalParm("seed"), len(sentences))
    path_info = _path_from_input(node)
    if path_info is not None:
        req["path"] = {"xz": path_info["xz"], "shares": path_info["shares"]}
        req["stem"] += "_path"
    payload = json.dumps(req).encode()
    box = {}

    def work():  # network only, no hou calls
        try:
            r = client._OPENER.open(urllib.request.Request(url + "/generate", payload,
                                                           {"Content-Type": "application/json"}), timeout=900)
            box["res"] = json.loads(r.read())
        except urllib.error.HTTPError as e:
            box["err"] = e.read().decode(errors="replace")[:1500]
        except Exception as e:
            box["err"] = "%s: %s" % (type(e).__name__, e)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    t0 = time.time()
    while th.is_alive():  # status bar only: an Esc must not abort a job the server finishes anyway
        hou.ui.setStatusMessage("Kimodo: generating '%s' ... %.0f s" % (prompt, time.time() - t0))
        th.join(0.3)
    if "err" in box:
        raise hou.Error("Kimodo server: " + box["err"])
    kp = box["res"]["kimodo_npz"].replace("\\", "/")
    side = os.path.splitext(kp)[0] + ".path.json"
    if path_info is not None:
        with open(side, "w") as fh:
            json.dump({"origin": path_info["origin"]}, fh)
        node.parm("inplace").set(0)
    elif os.path.exists(side):
        os.remove(side)
    node.parm("motion").set(kp)
    node.parm("use_motion").set(1)
    hou.ui.setStatusMessage("Kimodo: %d frames from '%s' in %.1f s" % (box["res"]["frames"], prompt, time.time() - t0))
