"""Human Generator server: SOMA-X / Anny bodies + NVIDIA Kimodo text-to-motion, kept warm on the GPU.

Houdini (or any other client) talks to it over HTTP on 127.0.0.1 and never imports torch.

Start:  server/start_server.bat   (Windows)  |  bash server/start_server.sh   (Linux, untested)
GET  /ping      -> {"ok": true, "device", "lods", "kimodo": "loading|ready|failed", ...}
POST /body      JSON:
  lod            "mid" | "low" | "xlo"                         (default "mid")
  backend        "anny" | "soma"                               (default "soma")
  units          {"age_years", "height_cm", "weight_kg"}       (anny: solved exactly)
  phenotype      {gender, muscle, proportions, cupsize, firmness, ...} 0..1   (anny)
  local_changes  {anny local change name: value}               (anny)
  identity       up to 128 PCA coefficients                    (soma)
  scales         60 bone-length ratios or {substring: ratio}   (soma)
  global_scale   float
  poses          optional (F, 77, 3) axis-angle; default = rest pose
  transl         optional (F, 3) hips translation in metres
  -> {"npz", "static", "cached", "ms", "anny": measured height/weight/age}
POST /generate  JSON: prompt (sentences), duration | durations, transition_frames, seed, stem,
                optional path {"xz": [[x, z], ...] relative to its start, "shares": [...]}
  -> {"kimodo_npz", "npz", "frames", "segments", "gen_s", "total_s"}

Units: metres, +Y up, +Z forward (Houdini's convention).
Environment: HG_PORT (8777), HG_DEVICE (cuda|cpu), HG_CACHE (<repo>/cache), KIMODO_MODEL (Kimodo-SOMA-RP-v1),
             KIMODO_PRELOAD (1), TEXT_ENCODER_DEVICE (cpu keeps the Llama 3 text encoder off the GPU).
"""
import hashlib
import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("HG_CACHE", ROOT / "cache"))
CACHE.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("HG_PORT", "8777"))
DEVICE = os.environ.get("HG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TOPK = 8  # skin influences kept per vertex for the client-side weights
MODELS_READY = CACHE / ".models_ready"  # written once Kimodo + its text encoder loaded: later starts run offline

from soma import SOMALayer  # noqa: E402

_layers = {}
_lock = Lock()


def _assets_dir(layer):
    for attr in ("data_root", "_data_root"):
        p = getattr(layer, attr, None)
        if p and (Path(p) / "SOMA_neutral.npz").exists():
            return Path(p)
    import soma
    for p in (Path(soma.__file__).resolve().parent.parent / "assets",
              Path(soma.__file__).resolve().parent / "assets"):
        if (p / "SOMA_neutral.npz").exists():
            return p
    raise FileNotFoundError("SOMA_neutral.npz not found (the SOMA-X assets download from Hugging Face on first use)")


def _write_static(layer, lod, path):
    """Topology, UVs, skeleton names/parents, top-K public skin weights."""
    z = np.load(_assets_dir(layer) / "SOMA_neutral.npz", allow_pickle=False)
    nV = int(layer(torch.zeros(1, 77, 3, device=DEVICE),
                   torch.zeros(1, layer.num_shape_components, device=DEVICE)).vertices.shape[1])
    out = {}
    if lod == "mid" and "face_vert_counts" in z.files:
        out["face_counts"] = z["face_vert_counts"].astype(np.int32)
        out["face_indices"] = z["face_vert_indices"].astype(np.int32)
        if "uv_indices_st" in z.files:
            out["corner_uv"] = z["uv_coord_st"][z["uv_indices_st"]].astype(np.float32)
    elif lod == "low" and "face_vert_counts_low" in z.files:
        out["face_counts"] = z["face_vert_counts_low"].astype(np.int32)
        out["face_indices"] = z["face_vert_indices_low"].astype(np.int32)
    if "face_counts" not in out or int(out["face_indices"].max()) >= nV:
        tri = layer.faces.detach().cpu().numpy().astype(np.int32).reshape(-1, 3)
        out = {"face_counts": np.full(len(tri), 3, np.int32), "face_indices": tri.reshape(-1)}

    W = layer.public_skinning_weights
    W = (W() if callable(W) else W).detach().cpu().numpy().astype(np.float32)  # (V, 78)
    idx = np.argsort(-W, axis=1)[:, :TOPK]
    w = np.take_along_axis(W, idx, 1)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-8)
    out["skin_idx"] = idx.astype(np.int32)
    out["skin_w"] = w.astype(np.float32)
    out["joint_names"] = np.array(layer.public_joint_names)
    out["joint_parents"] = layer.output_joint_parent_ids.detach().cpu().numpy().astype(np.int32)
    out["scale_names"] = np.array(layer.scale_param_names)
    out["num_vertices"] = np.int32(nV)
    np.savez(path, **out)


def get_layer(lod, backend="soma"):
    """backend 'soma' = native PCA identity (neutral), 'anny' = Anny phenotypes (gender, age, muscle, weight, height, proportions)."""
    key = (lod, backend)
    if key not in _layers:
        t = time.time()
        L = SOMALayer(identity_model_type=backend, device=DEVICE, lod=lod)
        if backend == "anny":
            A = L.identity_model.identity_model
            for extra in ("cupsize", "firmness"):
                if extra not in A.phenotype_labels:
                    A.phenotype_labels = A.phenotype_labels + [extra]
        _layers[key] = L
        static = CACHE / f"static_{lod}.npz"   # same topology / UVs / skin for both backends
        if not static.exists():
            _write_static(L, lod, static)
        print(f"[soma] layer {lod}/{backend} ready on {DEVICE} in {time.time() - t:.1f}s", flush=True)
    return _layers[key]


ANNY_PHENO = ("gender", "age", "muscle", "weight", "height", "proportions", "cupsize", "firmness")

# ---- real-world units for Anny: age in years, height in cm, weight in kg
_AGE_YEARS = [0.0, 1.0, 4.0, 11.0, 16.0, 18.0, 64.0, 110.0]       # Anny shape_calibration/boys.pth
_AGE_ANNY = [0.0, 0.05, 0.215, 0.415, 0.67, 0.77, 0.83, 1.0]
_solve_cache = {}
_last_info = {}


def years_to_anny_age(y):
    return float(np.interp(float(y), _AGE_YEARS, _AGE_ANNY))


def _anthro(L):
    if not hasattr(L, "_anthro"):
        from anny.anthropometry import Anthropometry
        L._anthro = Anthropometry(L.identity_model.identity_model.anny_model)
    return L._anthro


FAT_LABELS = ("measure-waist-circ-incr", "measure-hips-circ-incr", "measure-bust-circ-incr", "measure-thigh-circ-incr",
              "measure-upperarm-circ-incr", "measure-calf-circ-incr", "stomach-pregnant-incr", "neck-double-incr",
              "l-upperarm-fat-incr", "r-upperarm-fat-incr", "l-lowerarm-fat-incr", "r-lowerarm-fat-incr",
              "l-upperleg-fat-incr", "r-upperleg-fat-incr", "l-lowerleg-fat-incr", "r-lowerleg-fat-incr",
              "head-fat-incr", "buttocks-volume-incr", "measure-underbust-circ-incr", "measure-neck-circ-incr")


def _fat_changes(L, f):
    names = set(L.identity_model.identity_model.local_change_labels)
    return {k: float(f) for k in FAT_LABELS if k in names}


def _measure(L, ph, lc=None):
    A = L.identity_model.identity_model
    pk = {k: torch.tensor([float(ph[k])], device=DEVICE) for k in A.phenotype_labels}
    lck = {k: torch.tensor([float(v)], device=DEVICE) for k, v in (lc or {}).items()}
    with torch.no_grad():
        rv = A(phenotype_kwargs=pk, local_changes_kwargs=lck)
        am = _anthro(L)
        return float(am.height(rv)[0]), float(am.mass(rv)[0])


def _bisect(f, target, lo, hi, increasing=True, n=18):
    flo, fhi = f(lo), f(hi)
    if (target <= min(flo, fhi)) or (target >= max(flo, fhi)):
        return (lo if (target <= flo) == increasing else hi), False
    for _ in range(n):
        mid = (lo + hi) / 2
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, True


def anny_solve(L, ph, age_years=None, height_cm=None, weight_kg=None):
    """Find Anny phenotypes (+ a uniform scale) that give the requested age / height / weight.
    Returns (phenotype dict, global scale, info)."""
    key = json.dumps([ph, age_years, height_cm, weight_kg], sort_keys=True)
    if key in _solve_cache:
        return _solve_cache[key]
    ph = {k: float(ph.get(k, 0.5)) for k in ANNY_PHENO}
    if age_years is not None:
        ph["age"] = years_to_anny_age(age_years)
    scale, reached = 1.0, {"height": True, "weight": True}
    WMIN, WMAX = -0.5, 1.5      # mild extrapolation of the weight phenotype
    for _ in range(3):          # height and weight interact a little: alternate
        if height_cm is not None:
            def fh(x):
                q = dict(ph); q["height"] = x; return _measure(L, q)[0]
            ph["height"], reached["height"] = _bisect(fh, height_cm / 100.0, 0.0, 1.0)
        if weight_kg is not None:
            h_now = _measure(L, ph)[0]
            s = (height_cm / 100.0) / h_now if height_cm is not None else 1.0
            def fw(x):
                q = dict(ph); q["weight"] = x; return _measure(L, q)[1] * s ** 3
            ph["weight"], reached["weight"] = _bisect(fw, weight_kg, WMIN, WMAX)
    fat, lc = 0.0, {}
    if weight_kg is not None:
        h0, m0 = _measure(L, ph)
        s0 = (height_cm / 100.0) / h0 if height_cm is not None else 1.0
        if abs(m0 * s0 ** 3 - weight_kg) > 1.5:      # weight slider ran out: drive the fat local changes
            lo, hi = (0.0, 2.0) if weight_kg > m0 * s0 ** 3 else (-1.0, 0.0)
            def ff(x):
                q = _fat_changes(L, x); hh, mm = _measure(L, ph, q)
                ss = (height_cm / 100.0) / hh if height_cm is not None else 1.0
                return mm * ss ** 3
            fat, _ = _bisect(ff, weight_kg, lo, hi)
            lc = _fat_changes(L, fat)
    h, m = _measure(L, ph, lc)
    if height_cm is not None and h > 0:
        scale = (height_cm / 100.0) / h      # exact height (also covers 50 cm babies / 250 cm giants)
    info = {"height_cm": round(h * scale * 100, 1), "weight_kg": round(m * scale ** 3, 1),
            "age_years": age_years, "anny_age": round(ph["age"], 4), "scale": round(scale, 4),
            "weight_reached": bool(abs(m * scale ** 3 - (weight_kg or m * scale ** 3)) < 1.5),
            "phenotype": {k: round(v, 4) for k, v in ph.items()}, "fat": round(fat, 3)}
    info["local_changes"] = lc
    out = (ph, scale, info)
    if len(_solve_cache) > 256:
        _solve_cache.clear()
    _solve_cache[key] = out
    return out


def build_scales(L, scales):
    names = list(L.scale_param_names)
    s = np.ones(len(names), np.float32)
    if isinstance(scales, list):  # exact per-name ratios in scale_param_names order
        s[: len(scales)] = np.asarray(scales, np.float32)[: len(names)]
        return s
    for key, val in (scales or {}).items():
        for i, n in enumerate(names):
            if key in n:
                s[i] *= float(val)
    return s


def solve(req):
    lod = req.get("lod", "mid")
    backend = req.get("backend", "soma")
    L = get_layer(lod, backend)
    info = None
    if backend == "anny":
        ph = req.get("phenotype", {}) or {}
        u = req.get("units") or {}
        if u:
            ph, gs, info = anny_solve(L, ph, u.get("age_years"), u.get("height_cm"), u.get("weight_kg"))
            req["global_scale"] = float(req.get("global_scale", 1.0)) * gs
        ident = np.array([float(ph.get(k, 0.5)) for k in L.identity_model.identity_model.phenotype_labels], np.float32)
        names = list(L.scale_param_names)                                            # 139 Anny local changes, 0 = none
        scales = np.zeros(len(names), np.float32)
        merged = dict((info or {}).get("local_changes") or {})
        for k, v in (req.get("local_changes") or {}).items():
            merged[k] = merged.get(k, 0.0) + float(v)
        for k, v in merged.items():
            if k in names:
                scales[names.index(k)] = float(v)
    else:
        K = L.num_shape_components
        ident = np.zeros(K, np.float32)
        given = np.asarray(req.get("identity", []), np.float32)[:K]
        ident[: len(given)] = given
        scales = build_scales(L, req.get("scales"))
    poses = np.asarray(req.get("poses", np.zeros((1, 77, 3))), np.float32).reshape(-1, 77, 3)
    F = poses.shape[0]
    transl = req.get("transl")
    gscale = float(req.get("global_scale", 1.0))
    corr = bool(req.get("correctives", True))

    key = hashlib.sha1(json.dumps([lod, backend, ident.tolist(), scales.tolist(), gscale, corr,
                                   hashlib.sha1(poses.tobytes()).hexdigest(),
                                   transl], sort_keys=True).encode()).hexdigest()[:16]
    path = CACHE / f"body_{key}.npz"
    if info is not None:
        _last_info[str(path)] = info
    if path.exists():
        return path, lod, True

    dev = lambda a: torch.as_tensor(a, device=DEVICE)
    with torch.no_grad():
        out = L(dev(poses),
                dev(np.repeat(ident[None], F, 0)),
                scale_params=dev(np.repeat(scales[None], F, 0)),
                transl=None if transl is None else dev(np.asarray(transl, np.float32).reshape(F, 3)),
                apply_correctives=corr,
                global_scale=gscale)
        rest = L(dev(np.zeros((1, 77, 3), np.float32)), dev(ident[None]),
                 scale_params=dev(scales[None]), apply_correctives=False, global_scale=gscale)
    np.savez(path,
             verts=out.vertices.cpu().numpy().astype(np.float32),        # (F, V, 3)
             xforms=out.transforms.cpu().numpy().astype(np.float32),     # (F, 78, 4, 4) world
             rest_verts=rest.vertices[0].cpu().numpy().astype(np.float32),
             rest_xforms=rest.transforms[0].cpu().numpy().astype(np.float32))
    return path, lod, False


# ---------------------------------------------------------------- Kimodo (text -> motion), kept warm
import threading
KIMODO_MODEL = os.environ.get("KIMODO_MODEL", "Kimodo-SOMA-RP-v1")
MOTION_DIR = CACHE / "motions"
MOTION_DIR.mkdir(parents=True, exist_ok=True)
_kimodo = {"model": None, "state": "not loaded", "error": None}
_kimodo_lock = Lock()


def kimodo_model():
    with _kimodo_lock:
        if _kimodo["model"] is None:
            _kimodo["state"] = "loading"
            t = time.time()
            try:
                from kimodo import load_model
                _kimodo["model"] = load_model(KIMODO_MODEL, device=DEVICE if DEVICE != "cuda" else "cuda:0",
                                              default_family="Kimodo")
                _kimodo["state"] = "ready"
                MODELS_READY.write_text(KIMODO_MODEL)
                print(f"[kimodo] {KIMODO_MODEL} + text encoder loaded in {time.time() - t:.0f} s", flush=True)
            except Exception as e:
                _kimodo["state"], _kimodo["error"] = "failed", f"{type(e).__name__}: {e}"
                raise
        return _kimodo["model"]


def _mat_to_aa(R):
    R = np.asarray(R, np.float64)
    th = np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(th)[..., None]
    small = np.abs(s) < 1e-6
    aa = np.where(small, v / 2, v / np.where(small, 1, 2 * s) * th[..., None])
    for i in zip(*np.nonzero(th > np.pi - 1e-3)):
        M = R[i]; w = np.sqrt(np.clip((np.diag(M) + 1) / 2, 0, None)); k = int(np.argmax(w))
        a = M[:, k] + M[k, :]; a[k] = 2 * w[k] ** 2; aa[i] = a / np.linalg.norm(a) * th[i]
    return aa.astype(np.float32)


def _path_constraints(model, path, nfs):
    """path: {"xz": [[x, z], ...] polyline relative to its start, "shares": [fraction of the curve per segment] or None}.
    Returns ([Root2DConstraintSet], first_heading_angle). Kimodo heading 0 = facing +Z, angle = atan2(dx, dz)."""
    import torch
    from kimodo.constraints import Root2DConstraintSet
    P = np.asarray(path["xz"], np.float64)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-6]
    P = P[keep]
    L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    total = L[-1]
    shares = path.get("shares")
    if not shares or len(shares) != len(nfs):
        shares = [n / float(sum(nfs)) for n in nfs]          # same speed everywhere
    shares = np.asarray(shares, np.float64) / np.sum(shares)
    # per-frame speed: segment speed, eased in at the start and out at the end (~0.5 s)
    v = np.concatenate([np.full(n, s * total / n) for n, s in zip(nfs, shares)])
    T = len(v); r = min(15, T // 4)
    ramp = np.ones(T)
    ramp[:r] = np.linspace(0.15, 1.0, r)
    ramp[-r:] = np.minimum(ramp[-r:], np.linspace(1.0, 0.15, r))
    v = v * ramp
    d = np.r_[0.0, np.cumsum(v[:-1])]
    d = d / max(d[-1], 1e-9) * total
    xz = np.stack([np.interp(d, L, P[:, 0]), np.interp(d, L, P[:, 1])], 1)
    k = max(1, int(np.searchsorted(L, min(0.3, total))))
    dx, dz = P[k] - P[0]
    heading = float(np.arctan2(dx, dz))
    dev = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    c = Root2DConstraintSet(model.skeleton, torch.arange(T), torch.tensor(xz, dtype=torch.float32, device=dev))
    return [c], heading


def generate(req):
    """{prompt, duration, seed, inplace, stem, steps, identity?} -> kimodo npz + posed SOMA npz."""
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.tools import seed_everything
    model = kimodo_model()
    prompt = str(req["prompt"]).strip()
    texts = [x.strip() + "." for x in prompt.split(".") if x.strip()]
    durs = req.get("durations")                       # one duration (s) per sentence, else the same for all
    if durs and len(durs) == len(texts):
        nfs = [max(int(float(d) * model.fps), 10) for d in durs]
    else:
        nfs = [int(float(req.get("duration", 5.0)) * model.fps)] * len(texts)
    stem = req.get("stem") or "motion"
    t = time.time()
    with _lock:  # one GPU job at a time
        if req.get("seed") is not None:
            seed_everything(int(req["seed"]))
        cons, heading = [], None
        if req.get("path"):
            cons, heading = _path_constraints(model, req["path"], nfs)
        out = model(texts, nfs, constraint_lst=cons, num_denoising_steps=int(req.get("steps", 100)),
                    num_samples=1, multi_prompt=True, num_transition_frames=int(req.get("transition_frames", 5)),
                    post_processing=False, return_numpy=True, first_heading_angle=heading)
        single = {k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == 1 else v) for k, v in out.items()}
        kpath = MOTION_DIR / f"{stem}.npz"
        save_kimodo_npz(str(kpath), single)
        t_gen = time.time() - t
        z = np.load(kpath)
        poses = _mat_to_aa(z["local_rot_mats"])
        tr = z["root_positions"].astype(np.float32).copy()
        if req.get("inplace", True):
            tr[:, 0] = tr[0, 0]; tr[:, 2] = tr[0, 2]
        bpath, lod, _ = solve({"lod": req.get("lod", "mid"), "identity": req.get("identity", []),
                               "poses": poses.tolist(), "transl": tr.tolist(), "correctives": True})
    o = np.load(bpath)
    spath = MOTION_DIR / f"{stem}{'_inplace' if req.get('inplace', True) else ''}_soma.npz"
    np.savez(spath, verts=o["verts"], xforms=o["xforms"], static=np.array(str(CACHE / f"static_{lod}.npz")),
             posed_joints=z["posed_joints"], prompt=np.array(prompt))
    return {"npz": str(spath), "kimodo_npz": str(kpath), "frames": int(o["verts"].shape[0]), "segments": len(texts),
            "gen_s": round(t_gen, 1), "total_s": round(time.time() - t, 1)}


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/ping"):
            self._send(200, {"ok": True, "device": DEVICE, "lods": ["%s/%s" % k for k in _layers],
                             "kimodo": _kimodo["state"], "kimodo_error": _kimodo["error"]})
        else:
            self._send(404, {"error": "unknown path"})

    def do_POST(self):
        if self.path.startswith("/generate"):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                return self._send(200, generate(req))
            except Exception as e:
                traceback.print_exc()
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        if not self.path.startswith("/body"):
            return self._send(404, {"error": "unknown path"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            t = time.time()
            with _lock:
                path, lod, hit = solve(req)
            self._send(200, {"npz": str(path), "static": str(CACHE / f"static_{lod}.npz"),
                             "cached": hit, "ms": round((time.time() - t) * 1000), "anny": _last_info.get(str(path))})
        except Exception as e:  # report to the client instead of dying
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    warm = sys.argv[1:] or ["mid"]
    for lod in warm:
        get_layer(lod)
        solve({"lod": lod})  # compile Warp kernels now, not on the first slider move
    print(f"[soma] listening on http://127.0.0.1:{PORT}  (Ctrl+C to stop)", flush=True)
    if os.environ.get("KIMODO_PRELOAD", "1") == "1":  # load Kimodo + Llama 3 now, in the background
        threading.Thread(target=lambda: (print("[kimodo] preloading ...", flush=True), kimodo_model()), daemon=True).start()
    ThreadingHTTPServer.allow_reuse_address = False  # Windows would otherwise co-bind a busy port silently
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
