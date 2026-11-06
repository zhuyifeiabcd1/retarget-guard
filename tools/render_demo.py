"""Render the README demonstration from CC0 characters and procedural motion.

Everything here goes through the public package API: the motion is copied to the
heavy target and corrected frame by frame by RetargetGuard. The red overlay and
the depth label use an independent solid test (a limb vertex is inside when the
generalized winding number of the capped torso exceeds 0.5, and its depth is the
exact point-to-triangle distance), not the runtime penetration query.

Needs ffmpeg, Pillow and PyTorch, which the package itself does not require.
    python tools/render_demo.py --out media
"""
import argparse, ctypes, json, math, os, subprocess, tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from retarget_guard import RetargetGuard, vertex_normals
from retarget_guard.demo import belly_pose, demo_character, reach

HERE = Path(__file__).resolve().parent
BG, GREY = np.array((.20, .215, .24), np.float32), np.array((.80, .80, .82), np.float32)
LO, HI = np.array((1., .42, .38), np.float32), np.array((.62, 0., 0.), np.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FP, IP, UP = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ubyte)


def keyposes(source):
    d = source.data
    sem, v = d["semantic_map"], d["mesh_vertices"]
    pelvis = d["rest_joints"][sem[0], 1]
    h = np.ptp(v[:, 1])
    def surface(y):
        ring = v[(d["mesh_part"] == 0) & (np.abs(v[:, 1]-y) < .03*h)]
        return ring[:, 2].max(), np.abs(ring[:, 0]).max()
    chest_y, waist_y = pelvis+.25*h, pelvis+.09*h
    chest, _ = surface(chest_y)
    _, waist = surface(waist_y)
    crossed = np.zeros((source.nj, 3))
    reach(source, crossed, 0, np.array([-.062*h, chest_y-.085*h, chest+.12]), (.75, -.85, .05))
    reach(source, crossed, 1, np.array([.062*h, chest_y-.045*h, chest+.08]), (-.75, -.85, .05))
    hips = np.zeros((source.nj, 3))
    for side, sign in ((0, 1), (1, -1)):
        reach(source, hips, side, np.array([sign*(waist+.04), waist_y, .02]), (sign, .05, -.45))
    return dict(belly=belly_pose(source), crossed=crossed, hips=hips)


def motion(source, fps=30):
    """rest → key → rest for each key pose, with a slow breath-like spine sway."""
    k = keyposes(source)
    sem = source.data["semantic_map"]
    move, hold, rest = int(.8*fps), int(1.3*fps), int(.5*fps)
    seq, key = [], {}
    for name, pose in k.items():
        start = len(seq)
        a = np.linspace(0, 1, move); a = a*a*(3-2*a)
        seq += [x*pose for x in a]
        seq += [pose]*hold
        seq += [x*pose for x in a[::-1]]
        seq += [pose*0]*rest
        key[name] = (start, start+move+hold+move)
    m = np.stack(seq)
    t = np.arange(len(m))/fps
    sway = np.zeros_like(m)
    for j, amp in ((3, .012), (6, .012), (9, .012)):
        sway[:, sem[j], 2] = amp*np.sin(2*np.pi*.33*t)
        sway[:, sem[j], 0] = amp*.6*np.sin(2*np.pi*.21*t+1)
    return m+sway, key


CAPTION = {"belly": "Hands on the stomach: fine on the slim body, buried in the heavy one",
           "crossed": "Arms crossed: the forearms sink into the heavier chest",
           "hips": "The motion itself is broken: the hands are already inside the source body"}


def capped(faces, parts):
    tri = faces[(parts[faces] == 0).all(1)]
    nxt = {}
    for f in tri:
        for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            nxt.setdefault(int(u), set()).add(int(v))
    edge = {(u, v) for u, vs in nxt.items() for v in vs if u not in nxt.get(v, ())}
    loops = []
    while edge:
        a, b = edge.pop(); loop = [(a, b)]
        while b != a:
            e = next((x for x in edge if x[0] == b), None)
            if e is None:
                break
            edge.discard(e); loop.append(e); b = e[1]
        loops.append(loop)
    n, out, caps = len(parts), [tri], []
    for i, lp in enumerate(loops):
        caps.append(sorted({u for e in lp for u in e}))
        out.append(np.array([[v, u, n+i] for u, v in lp]))
    return np.concatenate(out), caps


def winding(q, a, b, c):
    out = []
    for qs in q.split(512):
        A, B, C = a[None]-qs[:, None], b[None]-qs[:, None], c[None]-qs[:, None]
        la, lb, lc = A.norm(dim=-1), B.norm(dim=-1), C.norm(dim=-1)
        num = (A*torch.cross(B, C, dim=-1)).sum(-1)
        den = la*lb*lc+(A*B).sum(-1)*lc+(B*C).sum(-1)*la+(C*A).sum(-1)*lb
        out.append(torch.atan2(num, den).sum(-1)/(2*np.pi))
    return torch.cat(out)


def distance(p, a, b, c):
    """Ericson closest point on a triangle. p [N,3], a,b,c [M,3] gives [N,M]."""
    p = p[:, None]; a, b, c = a[None], b[None], c[None]
    ab, ac, ap, bp, cp = b-a, c-a, p-a, p-b, p-c
    d1, d2, d3, d4, d5, d6 = ((x*y).sum(-1) for x, y in ((ab, ap), (ac, ap), (ab, bp), (ac, bp), (ab, cp), (ac, cp)))
    va, vb, vc = d3*d6-d5*d4, d5*d2-d1*d6, d1*d4-d3*d2
    den = 1/(va+vb+vc)
    r = a+ab*(vb*den)[..., None]+ac*(vc*den)[..., None]
    for m, e in (((va <= 0) & (d4 >= d3) & (d5 >= d6), b+(c-b)*((d4-d3)/((d4-d3)+(d5-d6)))[..., None]),
                 ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a+ac*(d2/(d2-d6))[..., None]),
                 ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a+ab*(d1/(d1-d3))[..., None]),
                 ((d6 >= 0) & (d5 <= d6), c), ((d3 >= 0) & (d4 <= d3), b), ((d1 <= 0) & (d2 <= 0), a)):
        r = torch.where(m[..., None], e, r)
    return ((p-r)**2).sum(-1).sqrt()


@torch.no_grad()
def inside_depth(v, T, caps, ids):
    """Depth in metres per vertex of ids inside the capped torso."""
    V = torch.as_tensor(v, dtype=torch.float32, device=DEV)
    ext = torch.cat([V]+[V[c].mean(0, keepdim=True) for c in caps])
    t = torch.as_tensor(T, device=DEV)
    a, b, c = ext[t[:, 0]], ext[t[:, 1]], ext[t[:, 2]]
    out = np.zeros(len(v), np.float32)
    i = torch.as_tensor(ids, device=DEV)
    i = i[((V[i] >= ext.amin(0)) & (V[i] <= ext.amax(0))).all(-1)]
    if len(i):
        i = i[winding(V[i], a, b, c) > .5]
    if len(i):
        d = torch.cat([distance(x, a, b, c).amin(-1) for x in V[i].split(64)])
        out[i.cpu().numpy()] = d.cpu().numpy()
    return out


def look_at(eye, target):
    z = (eye-target)/np.linalg.norm(eye-target)
    x = np.cross((0, 1, 0), z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4); m[0, :3], m[1, :3], m[2, :3] = x, y, z
    m[:3, 3] = -m[:3, :3]@eye
    return m


def camera(center, half, az):
    fov = math.radians(26)
    dist = half/math.tan(fov/2)*1.12
    az, el = math.radians(az), math.radians(8)
    d = np.array([math.sin(az)*math.cos(el), math.sin(el), math.cos(az)*math.cos(el)])
    f = 1/math.tan(fov/2)
    n, fr = max(.05, dist-2), dist+2
    proj = np.array([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, (fr+n)/(n-fr), 2*fr*n/(n-fr)], [0, 0, -1, 0]])
    return proj@look_at(center+d*dist, center), d


def draw(lib, v, faces, limb, depth, mvp, light, size):
    R = size*2
    hom = np.c_[v, np.ones(len(v))]@mvp.T
    px = np.stack(((hom[:, 0]/hom[:, 3]+1)/2*R, (1-hom[:, 1]/hom[:, 3])/2*R, hom[:, 3]), -1).astype(np.float32)
    t = np.clip((depth-.003)/.012, 0, 1)[:, None]
    n = vertex_normals(v, faces)
    lam = np.abs(n@light)[:, None]
    col = (np.where(depth[:, None] > .003, LO+(HI-LO)*t, GREY)*(.38+.62*lam)).astype(np.float32)
    hot = depth.astype(np.float32)
    out = np.zeros((R, R, 3), np.float32)
    f = faces.astype(np.int32); m = limb.astype(np.uint8)
    lib.raster(R, R, px.ctypes.data_as(FP), col.ctypes.data_as(FP), hot.ctypes.data_as(FP), len(f), f.ctypes.data_as(IP),
               m.ctypes.data_as(UP), ctypes.c_float(.003), BG.ctypes.data_as(FP), out.ctypes.data_as(FP))
    o = out.reshape(size, 2, size, 2, 3).mean((1, 3))
    return (np.clip(o, 0, 1)*255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("media"))
    ap.add_argument("--panel", type=int, default=420)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--lookahead", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        so = Path(tmp)/"soft_raster.so"
        subprocess.run([os.environ.get("CC", "cc"), "-O3", "-fPIC", "-shared", str(HERE/"soft_raster.c"), "-o", str(so)], check=True)
        lib = ctypes.CDLL(str(so))
        src, tgt = demo_character("source"), demo_character("target")
        m, key = motion(src, a.fps)
        if a.max_frames:
            m = m[:a.max_frames]
        ds, dt = src.data, tgt.data
        fs, ft = ds["mesh_faces"], dt["mesh_faces"]
        Ts, cs = capped(fs, ds["mesh_part"]); Tt, ct = capped(ft, dt["mesh_part"])
        arm_s = np.flatnonzero((ds["mesh_part"] >= 4) & (ds["mesh_part"] <= 7)); arm_t = np.flatnonzero((dt["mesh_part"] >= 4) & (dt["mesh_part"] <= 7))
        lim_s = ((ds["mesh_part"][fs] >= 2) & (ds["mesh_part"][fs] <= 7)).sum(1) >= 2
        lim_t = ((dt["mesh_part"][ft] >= 2) & (dt["mesh_part"][ft] <= 7)).sum(1) >= 2
        # Same scale in every panel, framed from the hips to the top of the head.
        vs, vt = ds["mesh_vertices"], dt["mesh_vertices"]
        center = lambda x: np.array([0, x[:, 1].max()-.40, .03])
        mvp_s, d0 = camera(center(vs), .47, 24); mvp_t, _ = camera(center(vt), .47, 24)
        light = d0*.8+np.array((0, .5, 0)); light /= np.linalg.norm(light)
        title = ["Source motion, slim body", "Copied to heavy body", "With Retarget Guard"]
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 18)
        small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 17)
        strip = 36
        W, H = a.panel*3, a.panel+strip
        where = np.empty(len(m), object)
        for n, (s, e) in key.items():
            where[s:min(e+int(.5*a.fps), len(m))] = n
        path = a.out/"demo.mp4"
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-",
                               "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        rows = []
        i = 0
        with RetargetGuard(tgt, src, lookahead=a.lookahead) as guard:
            def consume(pose):
                nonlocal i
                v0, v1 = tgt.skin(m[i]), tgt.skin(pose)
                v2 = guard.deform(v1, vertex_normals(v1, ft))
                s = src.skin(m[i])
                dep = [inside_depth(s, Ts, cs, arm_s), inside_depth(v0, Tt, ct, arm_t), inside_depth(v2, Tt, ct, arm_t)]
                ims = [draw(lib, s, fs, lim_s, dep[0], mvp_s, light, a.panel), draw(lib, v0, ft, lim_t, dep[1], mvp_t, light, a.panel), draw(lib, v2, ft, lim_t, dep[2], mvp_t, light, a.panel)]
                im = Image.new("RGB", (W, H), tuple(int(x*255*.72) for x in BG))
                im.paste(Image.fromarray(np.concatenate(ims, 1)), (0, strip)); w = ImageDraw.Draw(im)
                w.text((W/2, strip/2), CAPTION[where[i]], font=small, fill=(240, 240, 240), anchor="mm")
                r = []
                for k in range(3):
                    x = float(dep[k].max()*1000); r.append(x)
                    w.text((k*a.panel+10, strip+8), title[k], font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
                    w.text((k*a.panel+10, H-30), f"inside torso {x:5.1f} mm", font=font, fill=(255, 90, 80) if x > 10 else (235, 235, 235), stroke_width=2, stroke_fill=(0, 0, 0))
                ff.stdin.write(im.tobytes())
                rows.append(r); i += 1
                if i % 30 == 0:
                    print(i, len(m), np.round(r, 1), flush=True)
            for p in m:
                out = guard.step(p, source_pose=p)
                if out is not None:
                    consume(out)
            for out in guard.flush():
                consume(out)
        ff.stdin.close(); ff.wait()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vf", "fps=15,scale=840:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                        str(a.out/"demo.gif")], check=True)
        d = np.array(rows)
        rep = dict(frames=len(d), fps=a.fps, lookahead=a.lookahead, vertices=int(len(vt)), measure="capped-torso winding number with exact point-triangle distance",
                   max_mm=dict(zip(("source", "copy", "guard"), d.max(0).round(1).tolist())), mean_mm=dict(zip(("source", "copy", "guard"), d.mean(0).round(2).tolist())),
                   by_key={n: dict(zip(("source", "copy", "guard"), d[s:min(e, len(d))].max(0).round(1).tolist())) for n, (s, e) in key.items() if s < len(d)})
        json.dump(rep, open(a.out/"demo.json", "w"), indent=1)
        print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
