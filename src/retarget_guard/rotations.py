"""SO(3) conversions using NumPy only. Angles are radians."""
import numpy as np


def axis_angle_to_matrix(v):
    v = np.asarray(v, dtype=np.float64)
    if v.shape[-1:] != (3,) or not np.isfinite(v).all():
        raise ValueError("Expected finite rotation vectors [...,3]")
    t = np.linalg.norm(v, axis=-1)
    a = np.sinc(t / np.pi)
    b = .5 * np.sinc(t / (2 * np.pi)) ** 2
    x, y, z = np.moveaxis(v, -1, 0)
    zero = np.zeros_like(x)
    k = np.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(v.shape[:-1] + (3, 3))
    return np.eye(3) + a[..., None, None]*k + b[..., None, None]*(k @ k)


def matrix_to_axis_angle(r):
    r = np.asarray(r, dtype=np.float64)
    shape = r.shape[:-2]
    if r.shape[-2:] != (3, 3) or not np.isfinite(r).all():
        raise ValueError("Expected finite rotation matrices [...,3,3]")
    out = np.empty(shape + (3,))
    for m, dest in zip(r.reshape(-1, 3, 3), out.reshape(-1, 3)):
        skew = np.array([m[2, 1]-m[1, 2], m[0, 2]-m[2, 0], m[1, 0]-m[0, 1]]) * .5
        sine = np.linalg.norm(skew)
        angle = np.arctan2(sine, np.clip((np.trace(m)-1)*.5, -1, 1))
        if angle < 1e-7:
            dest[:] = skew * (1 + angle*angle/6)
        elif np.pi-angle < 1e-6:
            # Symmetric part has the rotation axis as its unit eigenvector.
            _, basis = np.linalg.eigh((m + m.T)*.5)
            axis = basis[:, -1]
            if np.dot(axis, skew) < 0:
                axis = -axis
            dest[:] = angle*axis
        else:
            dest[:] = skew * (angle/sine)
    return out


def as_matrices(pose, nj):
    p = np.asarray(pose, dtype=np.float64)
    if p.shape == (nj, 3):
        return axis_angle_to_matrix(p), True
    if p.shape != (nj, 3, 3) or not np.isfinite(p).all():
        raise ValueError(f"Expected finite pose [{nj},3] or [{nj},3,3]")
    if not np.allclose(p.swapaxes(-1, -2) @ p, np.eye(3), atol=1e-4) or (np.linalg.det(p) < 0).any():
        raise ValueError("Pose matrices must be proper rotations")
    return np.ascontiguousarray(p), False
