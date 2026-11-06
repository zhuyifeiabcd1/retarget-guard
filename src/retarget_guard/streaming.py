"""Single-frame streaming API with explicit lookahead latency."""
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .character import Character
from .rotations import as_matrices, matrix_to_axis_angle
from ._native import Pack, Guard, Settings


def frozen_settings(**overrides):
    """Return the h13 settings, optionally with explicit caller overrides."""
    config = json.loads(Path(__file__).with_name("settings.json").read_text())["settings"]
    config.update(overrides)
    return Settings(**config)


class RetargetGuard:
    """One stateful Layer 1 stream; one stateless Layer 2 deformer.

    step returns None for the first lookahead inputs, then the corrected oldest
    pose in its input representation. flush yields the remaining poses, without
    padding future frames. last_delta and last_stats belong to the latest output.
    source_pose is required when target and source use different skeletons or
    bind frames. The host supplies the upstream retargeted target pose.
    """
    def __init__(self, target, source, lookahead=0, fps=30, *, kappa=None, settings=None):
        if lookahead != int(lookahead) or not 0<=lookahead<=5:
            raise ValueError("lookahead must be an integer in 0..5")
        if not np.isfinite(fps) or fps<=0: raise ValueError("fps must be positive")
        if kappa is not None and (not np.isfinite(kappa) or not 0<=kappa<=1):
            raise ValueError("kappa must lie in [0,1]")
        self.target = target if isinstance(target,Character) else Character.load(target)
        self.source = source if isinstance(source,Character) else Character.load(source)
        self.lookahead, self.kappa = int(lookahead), kappa
        self.settings = replace(settings, fps=fps) if settings is not None else frozen_settings(fps=fps)
        self._target = self._source = self._guard = self._deformer = None
        self._queue = deque()
        self._finished = False
        self._closed = False
        self._failed = False
        self.last_delta = self.last_stats = self.last_deform_stats = None
        td, sd = self.target.data, self.source.data
        self._shared_pose = (self.target.nj==self.source.nj and
            all(np.array_equal(td[key],sd[key]) for key in ("parents","semantic_map")) and
            np.allclose(td["local_basis"],sd["local_basis"],atol=1e-6))
        try:
            self._target = Pack(self.target, cloth=kappa is not None)
            self._source = Pack(self.source, cloth=kappa is not None)
            self._guard = Guard(self._target,self._source,self.settings,kappa)
        except Exception:
            self.close()
            raise

    def _check(self):
        if self._closed: raise RuntimeError("Guard is closed")
        if self._failed: raise RuntimeError("Stream failed; create a new guard before continuing")

    def step(self, pose, *, source_pose=None):
        self._check()
        if self._finished: raise RuntimeError("Stream was flushed; create a new guard for a new clip")
        target, axis_angle = as_matrices(pose,self.target.nj)
        if source_pose is None:
            if not self._shared_pose:
                raise ValueError("Different skeletons/bind frames require source_pose")
            source = target
        else:
            source, _ = as_matrices(source_pose,self.source.nj)
        # Own buffered inputs: callers may reuse their frame buffers immediately.
        self._queue.append((target.copy(), source.copy(), axis_angle, np.asarray(pose).copy()))
        if len(self._queue)<=self.lookahead: return None
        return self._emit()

    def _emit(self):
        target, _, axis_angle, original = self._queue[0]
        try:
            delta, stats = self._guard.step(np.stack([x[0] for x in self._queue]),
                                            np.stack([x[1] for x in self._queue]))
        except Exception:
            self._failed = True
            raise
        self._queue.popleft()
        self.last_delta, self.last_stats = delta.copy(), stats.copy()
        result = self._guard.pose.copy()
        if axis_angle:
            result = matrix_to_axis_angle(result)
            # Keep all untouched joints in exactly the original representation.
            changed = self.target.data["semantic_map"][1:22][np.any(delta!=0,axis=-1)]
            out = original.astype(np.float64)
            out[changed] = result[changed]
            result = out
        return result

    def flush(self):
        """Yield buffered outputs in chronological order; inspect last_delta per yield."""
        self._check()
        self._finished = True
        while self._queue:
            yield self._emit()

    def deform(self, vertices, normals):
        """Return corrected final vertices after host skinning (same topology).

        Input normals must be outward unit normals of the posed mesh. Recompute
        normals after this operation before rendering. Cloth kappa affects Layer
        1 only; Layer 2 continues to deform body skin within its soft budget.
        """
        self._check()
        if self._deformer is None:
            from ._deform import Deformer
            d = self.target.data
            required = ("mesh_vertices","mesh_faces","mesh_part","mesh_soft")
            if any(key not in d for key in required):
                raise ValueError("Layer 2 needs full mesh fields; rebuild with build_character")
            self._deformer = Deformer(*(d[key] for key in required),d["pairs"],rho=self.settings.rho)
        displacement, self.last_deform_stats = self._deformer(vertices,normals)
        result = np.asarray(vertices,dtype=np.float32) + displacement
        if not np.isfinite(result).all(): raise FloatingPointError("Nonfinite deformed vertices")
        return result

    def close(self):
        if getattr(self,"_closed",False): return
        for obj in (getattr(self,"_deformer",None),getattr(self,"_guard",None),
                    getattr(self,"_source",None),getattr(self,"_target",None)):
            if obj is not None: obj.close()
        self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc): self.close()
    def __del__(self): self.close()
