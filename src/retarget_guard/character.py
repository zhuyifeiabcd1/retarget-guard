"""Offline mesh/skeleton preparation with no model or training dependency."""
from pathlib import Path
import warnings
import numpy as np
from .rotations import as_matrices

SEMANTICS = ("pelvis l_hip r_hip spine1 l_knee r_knee spine2 l_ankle r_ankle "
             "spine3 l_foot r_foot neck l_collar r_collar head l_shoulder "
             "r_shoulder l_elbow r_elbow l_wrist r_wrist").split()
PARTS = ("torso head l_uarm r_uarm l_farm r_farm l_hand r_hand "
         "l_thigh r_thigh l_shin r_shin l_foot r_foot").split()
BODY_PART = np.array([0,8,9,0,10,11,0,12,13,0,12,13,1,0,0,1,2,3,4,5,6,7])
Q_COUNT = [0,0,60,60,60,60,60,60,100,100,70,70,50,50]
B_COUNT = [1000,350,250,250,200,200,220,220,450,450,280,280,220,220]
MAX_DEPTH = np.array([.30,.13,.10,.10,.08,.08,.05,.05,.18,.18,.10,.10,.08,.08])
SOFT_F = [.25,.03,.35,.35,.25,.25,.10,.10,.35,.35,.20,.20,.10,.10]
SEGS = [[(0,3),(3,6),(6,9),(9,12),(9,13),(9,14)],[(12,15)],[(16,18)],[(17,19)],
        [(18,20)],[(19,21)],[(20,20)],[(21,21)],[(1,4)],[(2,5)],[(4,7)],[(5,8)],[(7,10)],[(8,11)]]


def _pairs():
    p = {name: i for i, name in enumerate(PARTS)}
    out = []
    for side, other in (("l", "r"), ("r", "l")):
        for arm in ("uarm", "farm", "hand"):
            out += [(p[f"{side}_{arm}"], 0, 0), (p[f"{side}_{arm}"], 1, 1)]
        for arm in ("farm", "hand"):
            for leg in ("l_thigh", "r_thigh", "l_shin", "r_shin"):
                out.append((p[f"{side}_{arm}"], p[leg], 2))
            for arm2 in ("farm", "hand"):
                out.append((p[f"{side}_{arm}"], p[f"{other}_{arm2}"], 3))
        for leg in ("shin", "foot"):
            for leg2 in ("thigh", "shin", "foot"):
                out.append((p[f"{side}_{leg}"], p[f"{other}_{leg2}"], 4))
            out.append((p[f"{side}_{leg}"], 0, 5))
    return np.array(out + [(8,9,6)], dtype=np.int64)


PAIRS = _pairs()


def vertex_normals(vertices, faces):
    """Area-weighted outward normals; faces must have outward winding."""
    v, f = np.asarray(vertices), np.asarray(faces)
    tri = v[f]
    fn = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:,k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


def soft_thickness(v, labels, joints):
    out = np.zeros(len(v), dtype=np.float32)
    for part, segments in enumerate(SEGS):
        ids = np.flatnonzero(labels == part)
        x = v[ids]
        distances = []
        for a, b in segments:
            origin, axis = joints[a], joints[b]-joints[a]
            u = np.clip(((x-origin)*axis).sum(-1)/np.maximum((axis*axis).sum(), 1e-9), 0, 1)
            distances.append(np.linalg.norm(x-(origin+u[:,None]*axis), axis=-1))
        out[ids] = SOFT_F[part] * np.min(distances, axis=0)
    return out


def _fps(points, count, seed=0):
    if len(points) <= count:
        return np.arange(len(points))
    ids = [int(np.random.default_rng(seed).integers(len(points)))]
    distance = np.linalg.norm(points-points[ids[0]], axis=1)
    for _ in range(count-1):
        i = int(distance.argmax())
        ids.append(i)
        distance = np.minimum(distance, np.linalg.norm(points-points[i], axis=1))
    return np.array(ids)


class Character:
    """Portable numeric pack. Full mesh fields enable Layer 2 and example LBS."""
    def __init__(self, data):
        self.data = dict(data)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(dict(z))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            np.savez_compressed(f, **self.data)
        return path

    @property
    def nj(self):
        return len(self.data["parents"])

    def skin(self, pose):
        """Reference host LBS for one frame; returns full mesh [V,3]."""
        d = self.data
        if "mesh_vertices" not in d:
            raise ValueError("Full mesh required for skinning; rebuild the character")
        r, _ = as_matrices(pose, self.nj)
        world, joints = np.empty_like(r), np.empty((self.nj,3))
        for j, parent in enumerate(d["parents"]):
            local = d["local_basis"][j] @ r[j]
            if parent < 0:
                world[j], joints[j] = local, d["local_offset"][j]
            else:
                world[j] = world[parent] @ local
                joints[j] = joints[parent] + world[parent] @ d["local_offset"][j]
        a = world @ d["rest_basis"].swapaxes(-1,-2)
        t = joints - (a @ d["rest_joints"][...,None])[...,0]
        wi, w, v = d["mesh_wi"], d["mesh_weights"], d["mesh_vertices"]
        return (((a[wi] @ v[:,None,:,None])[...,0] + t[wi]) * w[...,None]).sum(1)


def build_character(vertices, faces, skin_weights, joint_parents, rest_joints, semantic_map,
                    *, rest_basis=None, bone_parts=None, cloth_offset=None, output=None,
                    sampling_vertices=None, sampling_joints=None, sampling_weights=None,
                    max_bad_fraction=.01):
    """Build a humanoid pack in right-handed Y-up metres.

    semantic_map: 22 joint indices in SEMANTICS order, or name -> joint index.
    Extra joints inherit the part of their closest mapped ancestor. Optional
    rest_basis[J,3,3] contains world bind orientations. Sampling overrides keep
    vertex selection fixed across shape variants (used by from_smplh).
    Individual bad mesh rows are dropped and counted; >1% is an error by default.
    """
    v, f, w = np.asarray(vertices, np.float32), np.asarray(faces), np.asarray(skin_weights, np.float64)
    par, joints = np.asarray(joint_parents), np.asarray(rest_joints, np.float32)
    nj = len(par)
    if not 22 <= nj <= 128 or par.shape != (nj,) or joints.shape != (nj,3):
        raise ValueError("Expected 22..128 joints and rest_joints [J,3]")
    if not np.issubdtype(par.dtype, np.integer) or not np.isfinite(joints).all() or any(p < -1 or p >= j for j,p in enumerate(par)):
        raise ValueError("Expected finite joints and parent-before-child hierarchy")
    sem = np.array([semantic_map[name] for name in SEMANTICS] if isinstance(semantic_map, dict) else semantic_map)
    if sem.ndim != 1 or len(sem)<22 or not np.issubdtype(sem.dtype,np.integer):
        raise ValueError("Expected 22 integer body semantics")
    sem = sem[:22].astype(np.int64)
    if len(set(sem)) != 22 or (sem<0).any() or (sem>=nj).any():
        raise ValueError("Body semantics must map to distinct valid joints")
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or w.shape!=(len(v),nj):
        raise ValueError("Expected vertices [V,3] and skin_weights [V,J]")
    if f.ndim!=2 or f.shape[1]!=3 or not len(f) or not np.issubdtype(f.dtype,np.integer):
        raise ValueError("Expected triangle indices [F,3]")
    if not 0 <= max_bad_fraction < 1:
        raise ValueError("max_bad_fraction must lie in [0,1)")
    valid = np.isfinite(v).all(1) & np.isfinite(w).all(1) & (w>=0).all(1) & (w.sum(1)>0)
    valid_faces = (f>=0).all(1) & (f<len(v)).all(1)
    removed_v, removed_f = int((~valid).sum()), int((~valid_faces).sum())
    if removed_v > max_bad_fraction*len(v) or removed_f > max_bad_fraction*len(f):
        raise ValueError("Systematic invalid mesh vertices, weights or faces")
    f = f[valid_faces]
    keep_faces = valid[f].all(1)
    removed_f += int((~keep_faces).sum())
    if removed_f > max_bad_fraction*len(faces):
        raise ValueError("Systematic mesh face loss")
    ids = np.flatnonzero(valid)
    remap = np.full(len(v), -1); remap[ids] = np.arange(len(ids))
    f = remap[f[keep_faces]].astype(np.int64)
    v, w = v[valid], w[valid]
    if not len(f): raise ValueError("Mesh has no valid triangles")
    if removed_v or removed_f:
        warnings.warn(f"Skipped {removed_v} bad vertices and {removed_f} faces", RuntimeWarning)
    basis = np.tile(np.eye(3, dtype=np.float32), (nj,1,1)) if rest_basis is None else np.asarray(rest_basis,np.float32)
    if basis.shape!=(nj,3,3) or not np.isfinite(basis).all() or not np.allclose(basis.swapaxes(-1,-2) @ basis,np.eye(3),atol=1e-4) or (np.linalg.det(basis)<0).any():
        raise ValueError("Expected proper world rest_basis [J,3,3]")
    lb, lo = np.empty_like(basis), np.empty_like(joints)
    for j,p in enumerate(par):
        lb[j] = basis[j] if p<0 else basis[p].T @ basis[j]
        lo[j] = joints[j] if p<0 else basis[p].T @ (joints[j]-joints[p])
    if bone_parts is None:
        mapped = dict(zip(sem, BODY_PART))
        bp = np.zeros(nj, dtype=np.int64)
        for j,p in enumerate(par):
            bp[j] = mapped.get(j, bp[p] if p>=0 else 0)
    else:
        bp = np.asarray(bone_parts)
        if bp.shape!=(nj,) or not np.issubdtype(bp.dtype,np.integer) or (bp<0).any() or (bp>=14).any():
            raise ValueError("bone_parts must contain labels 0..13 for each joint")
    sw = w if sampling_weights is None else np.asarray(sampling_weights,np.float64)[valid]
    if sw.shape!=w.shape or not np.isfinite(sw).all() or (sw<0).any() or (sw.sum(1)<=0).any():
        raise ValueError("Invalid sampling_weights")
    pw = np.zeros((len(v),14))
    for j,p in enumerate(bp): pw[:,p] += sw[:,j]
    pw /= sw.sum(1,keepdims=True)
    labels, purity = pw.argmax(1), pw.max(1)
    sv = v if sampling_vertices is None else np.asarray(sampling_vertices)[valid]
    sj = joints[sem] if sampling_joints is None else np.asarray(sampling_joints)
    if sv.shape!=v.shape or sj.shape!=(22,3) or not np.isfinite(sv).all() or not np.isfinite(sj).all():
        raise ValueError("Invalid sampling coordinates")
    qi, bi = [], []
    for p,name in enumerate(PARTS):
        if Q_COUNT[p]:
            sel = np.flatnonzero((labels==p)&(purity>=.9))
            if p in (2,3):
                a,b = (16,18) if p==2 else (17,19)
                axis = sj[b]-sj[a]
                if axis@axis < 1e-12: raise ValueError("Zero-length upper arm")
                sel = sel[((sv[sel]-sj[a])@axis/(axis@axis))>.35]
            if not len(sel): raise ValueError("No high-purity query surface for " + name)
            qi.extend(sel[_fps(sv[sel], Q_COUNT[p])])
        sel = np.flatnonzero((labels==p)&(purity>=.6))
        if not len(sel): raise ValueError("No collider surface for " + name)
        bi.extend(sel[_fps(sv[sel],B_COUNT[p],seed=1)])
    qi, bi = np.asarray(qi), np.asarray(bi)
    selected = np.r_[qi,bi]
    wi = np.argsort(-w,axis=1,kind="stable")[:,:4]
    weights = np.take_along_axis(w,wi,1)
    weights = (weights/weights.sum(1,keepdims=True)).astype(np.float32)
    n = vertex_normals(v,f)
    soft = soft_thickness(v,labels,joints[sem])
    data = dict(parents=par.astype(np.int64),semantic_map=sem,local_basis=lb,local_offset=lo,
                rest_basis=basis,rest_joints=joints,vertices=v[selected],normals=n[selected],
                wi=wi[selected],weights=weights[selected],soft=soft[selected],part=labels[selected],
                query=np.r_[np.ones(len(qi)),np.zeros(len(bi))].astype(np.int32),
                vertex_ids=ids[selected],q_indices=np.arange(len(qi)),pairs=PAIRS.copy(),max_depth=MAX_DEPTH.copy(),
                mesh_vertices=v,mesh_faces=f,mesh_wi=wi,mesh_weights=weights,mesh_part=labels,mesh_soft=soft,
                dropped_vertices=np.array(removed_v),dropped_faces=np.array(removed_f))
    if cloth_offset is not None:
        offset = np.asarray(cloth_offset,np.float64)
        if offset.shape!=(len(valid),) or not np.isfinite(offset).all() or (offset<0).any():
            raise ValueError("cloth_offset must contain finite nonnegative metres per input vertex")
        data["cloth_offset"] = offset[valid][selected]
    result = Character(data)
    if output is not None: result.save(output)
    return result


def from_smplh(model_npz_path, betas, *, output=None):
    """Build from a user-owned SMPL-H NPZ; never downloads or bundles models.

    Required numeric fields: v_template, shapedirs, J_regressor (dense),
    weights, f (or faces), kintree_table. Pose blendshapes are not applied.
    """
    with np.load(model_npz_path,allow_pickle=False) as z:
        vt, sd = np.asarray(z["v_template"],np.float32), np.asarray(z["shapedirs"],np.float32)
        b = np.asarray(betas,np.float32).reshape(-1)
        if len(b)!=sd.shape[-1] or not np.isfinite(b).all():
            raise ValueError("betas dimension must match shapedirs and be finite")
        v = vt + (sd*b).sum(-1)
        jr = np.asarray(z["J_regressor"],np.float32)
        joints = jr @ v
        if len(joints)!=52: raise ValueError("from_smplh expects 52 joints")
        par = z["kintree_table"][0].astype(np.int64); par[0] = -1
        # Reference sampling is defined on the model template in float64.
        sample_v = np.asarray(z["v_template"],np.float64)
        sample_j = np.asarray(z["J_regressor"],np.float64) @ sample_v
        return build_character(v,z["f"] if "f" in z else z["faces"],z["weights"],par,joints,np.arange(52),
                               output=output,sampling_vertices=sample_v,sampling_joints=sample_j[:22])
