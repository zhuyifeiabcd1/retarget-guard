"""Private ctypes adapter. C configuration order matches the research solver."""
import ctypes as C
from dataclasses import dataclass, asdict
import numpy as np

from ._build import library_path
DP = C.POINTER(C.c_double)
IP = C.POINTER(C.c_int)

def _ptr(a):
    return a.ctypes.data_as(IP if a.dtype == np.int32 else DP)

def _lib(cloth=False):
    lib = C.CDLL(str(library_path("guard_core_cloth" if cloth else "guard_core")))
    lib.guard_pack.argtypes = [C.c_int]*3 + [IP]*2 + [DP]*5 + [IP] + [DP]*3 + [IP]*3 + [DP]
    if cloth:
        lib.guard_pack.argtypes += [DP]
    lib.guard_pack.restype = C.c_void_p
    lib.guard_free_pack.argtypes = [C.c_void_p]
    lib.guard_free_pack.restype = None
    lib.guard_create.argtypes = [C.c_void_p, C.c_void_p, DP]
    lib.guard_create.restype = C.c_void_p
    lib.guard_free.argtypes = [C.c_void_p]
    lib.guard_free.restype = None
    lib.guard_step.argtypes = [C.c_void_p, DP, DP, C.c_int, DP, DP, DP]
    lib.guard_geometry.argtypes = [C.c_void_p, DP, DP] + [C.c_double]*(2 if cloth else 1) + [DP]*5
    lib.guard_jacobian.argtypes = [C.c_void_p, DP, DP, C.c_int, C.c_int, DP, DP]
    lib.guard_box.argtypes = [C.c_int]*3 + [DP]*8
    lib.guard_set_delta.argtypes = [C.c_void_p, DP]
    lib.guard_clearance.argtypes = [C.c_void_p, DP]
    return lib

@dataclass
class Settings:
    K: int = 48
    sweeps: int = 8
    w_vel: float = 10000.
    w_acc: float = 45000.
    w_pen: float = 10000000.
    fps: float = 30.
    zero_step: float = 0.
    xi: float = 0.
    influence: float = .02
    margin: float = 0. # unused sphere-layout slot; must be zero
    rho: float = .5
    w_hand: float = 1000000.
    w_hand_rot: float = 50.
    w_foot: float = 30.
    w_contact: float = 1000000.
    spring_hz: float = 0.
    eps: float = .002
    fade: float = 1.
    pull_travel: float = .06
    row_alpha: float = 1.
    calibration_scale: float = 0.
    source_scale: float = 0.
    positive_offsets: float = 0.
    w_pos_vel: float = 1000000.
    w_pos_acc: float = 1000000.

    w_res: float = 100.
    spine_cost: float = 1.
    spine_limit: float = 3.
    neck_limit: float = 3.
    hip_limit: float = 8.
    wrist_limit: float = 8.
    arm_limit: float = 15.
    leg_limit: float = 8.

    density: int = 1  # sparse / frame-adaptive / full
    dense_near: float = .02
    limit_gain: float = 3.
    max_speed: float = 30.  # degrees/second, joint norm envelope
    max_acc: float = 90.  # degrees/second squared
    slack_cap: float = .01  # bounded collision force for deep conflicts
    hand_relax: float = 2.
    clearance_band: float = 10.  # shared deadband for conservative rest AABBs
    release_tau: float = 0.  # frames; joints no row touches decay to zero at this time constant
    snap_deg: float = 0.     # joints no row touches and below this go to exact zero
    source_allow: float = 0.  # 0: the model's own penetration is corrected too; 1: it is tolerated
    rho_hand: float = .1      # soft budget for a hand pressing on anything
    rho_forearm: float = .1   # soft budget for a forearm pressing on the torso
    settle: int = 12          # solver steps on the first frame before the first output
    urg_d0: float = 0.        # metres; hand or forearm into torso beyond this starts urgency
    urg_d1: float = 0.        # full urgency here; d1 <= d0 disables it
    urg_tau: float = 15.      # frames; urgency decay
    urg_smooth: float = 1.    # smoothing weights divided by this at full urgency
    urg_speed: float = 1.     # speed limit multiplied by this at full urgency
    urg_acc: float = 1.       # acceleration limit multiplied by this at full urgency
    urg_slack: float = 1.     # deep-row force cap multiplied by this at full urgency

    def array(self):
        if self.density not in (0,1,2) or not 0 <= self.dense_near <= .04:
            raise ValueError('Invalid density mode or distance')
        if bool(self.max_speed) != bool(self.max_acc):
            raise ValueError('Velocity and acceleration bounds must be enabled together')
        if self.max_acc and self.zero_step:
            raise ValueError('Exact-zero shrinkage is incompatible with acceleration bounds')
        if self.sweeps != int(self.sweeps) or not 1 <= self.sweeps <= 64:
            raise ValueError('sweeps must be an integer in 1..64')
        if self.K != int(self.K) or not 1<=self.K<=128:raise ValueError('K must be an integer in 1..128')
        if not 0 <= self.source_allow <= 1 or self.settle != int(self.settle) or not 0 <= self.settle <= 64:raise ValueError('Invalid source_allow or settle')
        if self.calibration_scale or self.source_scale or self.positive_offsets or self.margin:raise ValueError('Mesh queries have no sphere calibration offsets')
        return np.array(list(asdict(self).values()), dtype=np.float64)

class Pack:
    def __init__(self, character, cloth=False):
        self.cloth = cloth
        self.lib = _lib(cloth)
        self.data = character.data
        semantic_map = self.data["semantic_map"]
        d = self.data
        self.nj, self.ns, self.np = len(d['parents']), len(d['vertices']), len(d['pairs'])
        sem = np.asarray(semantic_map, dtype=np.int32)[:22]
        if sem.shape != (22,) or len(set(sem)) != 22:
            raise ValueError('21 independent body controls and root must have distinct semantic joints')
        self.semantic_map = sem
        keys = ['parents', 'local_basis', 'local_offset', 'rest_basis', 'rest_joints', 'vertices', 'wi', 'weights', 'normals', 'soft', 'part', 'query', 'pairs']
        arrays = {k:np.ascontiguousarray(d[k], dtype=np.int32 if k in ('parents','wi','part','query','pairs') else np.float64) for k in keys}
        expected = dict(parents=(self.nj,), local_basis=(self.nj,3,3), local_offset=(self.nj,3), rest_basis=(self.nj,3,3), rest_joints=(self.nj,3), vertices=(self.ns,3),wi=(self.ns,4),weights=(self.ns,4),normals=(self.ns,3),soft=(self.ns,),part=(self.ns,),query=(self.ns,),pairs=(self.np,3))
        if any(arrays[k].shape != expected[k] or not np.isfinite(arrays[k]).all() for k in keys):
            raise ValueError('Invalid pack array shapes or values')
        args = [arrays['parents'], sem] + [arrays[k] for k in keys[1:]]
        depth=np.ascontiguousarray(d['max_depth'],dtype=np.float64)
        if depth.shape!=(14,) or not np.isfinite(depth).all() or (depth<=0).any():raise ValueError('Invalid maximum depths')
        extra = []
        if cloth:
            offset = np.ascontiguousarray(d.get("cloth_offset", np.zeros(self.ns)), dtype=np.float64)
            if offset.shape != (self.ns,) or not np.isfinite(offset).all() or (offset < 0).any():
                raise ValueError("Invalid cloth offsets")
            extra = [_ptr(offset)]
        self.handle = self.lib.guard_pack(self.nj, self.ns, self.np, *map(_ptr,args), _ptr(depth), *extra)
        if not self.handle:
            raise ValueError('Invalid character pack: dimensions, hierarchy, weights, parts, or semantics')

    def clearance(self):
        out=np.empty(2);self.lib.guard_clearance(self.handle,_ptr(out));return out

    def close(self):
        if getattr(self, 'handle', None):
            self.lib.guard_free_pack(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def geometry(self, pose, delta=None, rho=.5, kappa=1.):
        if not self.handle:
            raise RuntimeError("Character pack is closed")
        pose = np.ascontiguousarray(pose, dtype=np.float64)
        delta = np.zeros((21,3)) if delta is None else np.ascontiguousarray(delta,dtype=np.float64)
        if pose.shape != (self.nj,3,3) or delta.shape != (21,3) or not np.isfinite(pose).all() or not np.isfinite(delta).all() or not np.isfinite(rho) or rho<0:
            raise ValueError('Geometry pose/delta shape mismatch')
        out = [np.empty(sh) for sh in [(self.ns,3),(self.nj,3),(self.nj,3,3),(self.np,),(self.np,)]]
        extra = [kappa] if self.cloth else []
        rc = self.lib.guard_geometry(self.handle,_ptr(pose),_ptr(delta),rho,*extra,*map(_ptr,out))
        if rc: raise RuntimeError(f'C geometry failed: {rc}')
        return out

class Guard:
    def __init__(self, target, source, settings=None, kappa=None):
        self.target, self.source = target, source
        self.settings = settings or Settings()
        if self.settings.influence+self.settings.eps>.04:raise ValueError('Active range must fit the 40 mm positive search cutoff')
        self.lib = target.lib
        self.pose = np.empty((target.nj,3,3))
        cfg = self.settings.array()
        if kappa is not None:
            cfg = np.r_[cfg, kappa]
        self.handle = self.lib.guard_create(target.handle, source.handle, _ptr(cfg))
        if not self.handle: raise ValueError('Invalid settings or mismatched source/target pair semantics')

    def close(self):
        if getattr(self,'handle',None):
            self.lib.guard_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def step(self, target_window, source_window):
        if not self.handle:
            raise RuntimeError("Guard is closed")
        a = np.ascontiguousarray(target_window,dtype=np.float64)
        b = np.ascontiguousarray(source_window,dtype=np.float64)
        horizon = len(a)-1
        if a.shape != (horizon+1,self.target.nj,3,3) or b.shape != (horizon+1,self.source.nj,3,3) or not 0 <= horizon <= 5:
            raise ValueError('Require matching pose windows with 0..5 future frames')
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite input pose')
        out, stats = np.empty((21,3)), np.empty(22)
        rc = self.lib.guard_step(self.handle,_ptr(a),_ptr(b),horizon,_ptr(out),_ptr(self.pose),_ptr(stats))
        if rc: raise RuntimeError(f'C guard step failed: {rc}')
        if not np.isfinite(out).all() or not np.isfinite(self.pose).all():
            raise FloatingPointError("Nonfinite C guard output")
        return out, stats
