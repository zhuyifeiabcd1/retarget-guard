"""Layer 2, skin contact deformer. Runs after skinning on the final vertices.

NumPy/ctypes adapter; no Torch dependency. See deform_core.c for the C ABI.

The default collider set is the rendered mesh itself. A separate collider_mesh
(rest vertices, faces, parts, soft thickness) can retain detailed original hands
while the rendered mesh uses another tessellation. Pass both posed collider
vertices and outward unit normals to each call in that case. No rig, skinning
or history is needed at runtime. One instance per thread.

Part labels follow retarget_guard.PARTS (hands 6/7, arms 2..5). Detection uses actual
surface patches: 8 mm seeds for hand/hand, 22 mm for hands against other parts,
40 mm elsewhere. Each patch queries the opposing full vertex set, then refines
contact using every actual posed vertex in that patch. rho*soft is a
hard displacement budget. Open part boundaries retain the original seam mask.
"""
import ctypes as C
import heapq
import numpy as np

from ._build import library_path
FP, IP, DP = C.POINTER(C.c_float), C.POINTER(C.c_int), C.POINTER(C.c_double)


def ptr(a):
    return a.ctypes.data_as({np.dtype('float32'): FP, np.dtype('int32'): IP,
                            np.dtype('float64'): DP}[a.dtype])


def prepare(rest, faces, parts, pairs):
    """Static feathered part membership and 6 cm shared-seam exclusion."""
    nv, npart = len(rest), int(parts.max())+1
    edges = np.unique(np.sort(np.concatenate([faces[:,[0,1]], faces[:,[1,2]],
                                             faces[:,[2,0]]]), axis=1), axis=0)
    edge = np.concatenate([edges, edges[:,::-1]])
    pw = np.eye(npart, dtype=np.float32)[parts]
    degree = np.maximum(np.bincount(edge[:,0], minlength=nv), 1)
    for _ in range(8):
        accum = np.zeros_like(pw)
        np.add.at(accum, edge[:,0], pw[edge[:,1]])
        pw = .5*pw + .5*accum/degree[:,None]
    adjacency = [[] for _ in range(nv)]
    for (a,b), length in zip(edges, np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]], axis=1)):
        adjacency[a].append((int(b), float(length))); adjacency[b].append((int(a), float(length)))
    masks = np.ones((nv,len(pairs)), dtype=np.float32)
    for k,(a,b) in enumerate(pairs):
        boundary = edges[((parts[edges[:,0]]==a)&(parts[edges[:,1]]==b)) |
                         ((parts[edges[:,0]]==b)&(parts[edges[:,1]]==a))]
        dist = np.full(nv, .06)
        seeds = np.unique(boundary); dist[seeds] = 0
        queue = [(0.,int(i)) for i in seeds]; heapq.heapify(queue)
        while queue:
            d,i = heapq.heappop(queue)
            if d > dist[i]: continue
            for j,w in adjacency[i]:
                if d+w < dist[j]:
                    dist[j]=d+w; heapq.heappush(queue,(d+w,j))
        t = dist/.06
        masks[:,k] = t**3*(10-15*t+6*t*t)
    # Actual posed vertex seeds, with denser coverage on thin hands. Full
    # collider surfaces stay in the nearest-vertex grids.
    query = np.zeros(nv, dtype=np.int32)
    for p in range(npart):
        ids = np.flatnonzero(parts==p)
        for bit,spacing in ((2,.022 if p in (6,7) else .040),
                            (4,.008 if p in (6,7) else .030)):
            available = np.ones(len(ids), dtype=bool)
            for k,i in enumerate(ids):
                if not available[k]: continue
                query[i] |= bit
                available[np.linalg.norm(rest[ids]-rest[i],axis=1)<spacing] = False
    group = np.arange(nv, dtype=np.int32)
    for p in range(npart):
        ids=np.flatnonzero(parts==p); seeds=ids[(query[ids]&2)>0]
        if len(seeds):
            group[ids]=seeds[np.argmin(((rest[ids,None]-rest[seeds])**2).sum(-1),axis=1)]
    thin_group=np.arange(nv,dtype=np.int32)
    for p in range(npart):
        ids=np.flatnonzero(parts==p);seeds=ids[(query[ids]&4)>0]
        if len(seeds):thin_group[ids]=seeds[np.argmin(((rest[ids,None]-rest[seeds])**2).sum(-1),axis=1)]
    return np.ascontiguousarray(pw, dtype=np.float32), masks, query, group, thin_group


class Deformer:
    def __init__(self, rest_vertices, faces, parts, soft, pairs, rho=.5, rim=.035, collider_mesh=None,
                 part_weights=None, pair_masks=None):
        v=np.ascontiguousarray(rest_vertices,dtype=np.float32)
        f=np.ascontiguousarray(faces,dtype=np.int32)
        p=np.ascontiguousarray(parts,dtype=np.int32)
        s=np.ascontiguousarray(soft,dtype=np.float32)
        pairs=np.asarray(sorted({tuple(sorted(map(int,x[:2]))) for x in pairs if x[0]!=x[1]}), dtype=np.int32).reshape(-1,2)
        if v.ndim!=2 or v.shape[1]!=3 or p.shape!=(len(v),) or s.shape!=p.shape or not len(v):
            raise ValueError('Expected vertices [V,3], parts and soft [V].')
        if f.ndim!=2 or f.shape[1]!=3 or not len(f) or f.min()<0 or f.max()>=len(v):
            raise ValueError('Invalid triangle indices.')
        if not np.isfinite(v).all() or not np.isfinite(s).all() or (s<0).any() or p.min()<0 or p.max()>=32:
            raise ValueError('Invalid rest vertices, part labels or soft thickness.')
        if len(pairs) and (pairs.min()<0 or pairs.max()>p.max()): raise ValueError('Pair label outside mesh.')
        pw,mask,query,group,thin_group=prepare(v,f,p,pairs)
        # Material attributes should be interpolated when subdividing a mesh,
        # just like skin weights; re-diffusing labels changes the material.
        if part_weights is not None:
            pw=np.ascontiguousarray(part_weights,dtype=np.float32)
            if pw.shape!=(len(v),int(p.max())+1) or not np.isfinite(pw).all() or (pw<0).any() or not np.allclose(pw.sum(1),1,atol=1e-5):
                raise ValueError('Expected normalized part weights [V,P].')
        if pair_masks is not None:
            mask=np.ascontiguousarray(pair_masks,dtype=np.float32)
            if mask.shape!=(len(v),len(pairs)) or not np.isfinite(mask).all() or (mask<0).any() or (mask>1).any():
                raise ValueError('Expected pair seam masks [V,pairs] in [0,1].')
        cp,cs,cmask=p,s,mask
        self.nc=len(v)
        if collider_mesh is not None:
            cv,cf,cp,cs=collider_mesh
            cv=np.ascontiguousarray(cv,dtype=np.float32);cf=np.ascontiguousarray(cf,dtype=np.int32)
            cp=np.ascontiguousarray(cp,dtype=np.int32);cs=np.ascontiguousarray(cs,dtype=np.float32)
            self.nc=len(cv)
            if cv.shape!=(self.nc,3) or cp.shape!=(self.nc,) or cs.shape!=cp.shape or not np.isfinite(cv).all():
                raise ValueError('Invalid collider mesh arrays.')
            if not len(cf) or cf.ndim!=2 or cf.shape[1]!=3 or cf.min()<0 or cf.max()>=self.nc or cp.min()<0 or cp.max()>p.max():
                raise ValueError('Invalid collider faces or parts.')
            _,cmask,query,group,thin_group=prepare(cv,cf,cp,pairs)
        self.nv,self.npart=len(v),int(p.max())+1
        self.faces,self.parts,self.soft,self.pairs=f,p,s,pairs
        self.part_weights,self.pair_masks=pw,mask
        self.lib=C.CDLL(str(library_path("deform_core")))
        self.lib.deform_create.argtypes=[C.c_int]*5+[IP]*7+[FP]*5+[C.c_float]*2
        self.lib.deform_create.restype=C.c_void_p
        self.lib.deform_free.argtypes=[C.c_void_p]
        self.lib.deform_free.restype=None
        self.lib.deform_step.argtypes=[C.c_void_p]+[FP]*5+[DP]
        self.handle=self.lib.deform_create(self.nv,len(f),self.nc,len(pairs),self.npart,
            ptr(f),ptr(p),ptr(pairs),ptr(cp),ptr(query),ptr(group),ptr(thin_group),ptr(s),ptr(pw),ptr(mask),ptr(cs),ptr(cmask),rho,rim)
        if not self.handle: raise ValueError('Invalid deformation configuration or allocation failure.')

    def close(self):
        if getattr(self,'handle',None):
            self.lib.deform_free(self.handle);self.handle=None

    def __del__(self): self.close()

    def __call__(self, vertices, normals, out=None, collider_vertices=None, collider_normals=None):
        if not self.handle: raise RuntimeError("Deformer is closed")
        v,n=(np.ascontiguousarray(a,dtype=np.float32) for a in (vertices,normals))
        if v.shape!=(self.nv,3) or n.shape!=v.shape: raise ValueError('Posed vertices/normals shape mismatch.')
        cv=v if collider_vertices is None else np.ascontiguousarray(collider_vertices,dtype=np.float32)
        cn=n if collider_normals is None else np.ascontiguousarray(collider_normals,dtype=np.float32)
        if cv.shape!=(self.nc,3) or cn.shape!=cv.shape:raise ValueError('Posed colliders/normals shape mismatch.')
        if out is None: out=np.empty_like(v)
        if out.shape!=v.shape or out.dtype!=np.float32 or not out.flags.c_contiguous or np.shares_memory(out,v) or np.shares_memory(out,n):
            raise ValueError('Output must be a separate contiguous float32 [V,3] buffer.')
        stats=np.empty(8,dtype=np.float64)
        if np.shares_memory(out,cv) or np.shares_memory(out,cn):raise ValueError('Output must not alias colliders.')
        rc=self.lib.deform_step(self.handle,ptr(v),ptr(n),ptr(cv),ptr(cn),ptr(out),ptr(stats))
        if rc: raise RuntimeError(f'C deformation failed: {rc}')
        if not np.isfinite(out).all(): raise FloatingPointError('Nonfinite C deformation output')
        return out,stats

