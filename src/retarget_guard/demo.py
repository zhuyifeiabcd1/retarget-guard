"""CC0 MPFB characters and a deterministic, procedural belly-touch motion."""
from pathlib import Path
import numpy as np
from .character import build_character
from .rotations import axis_angle_to_matrix, matrix_to_axis_angle


def demo_character(which="target"):
    if which not in ("source", "target"):
        raise ValueError("Choose source or target")
    with np.load(Path(__file__).with_name("data") / f"demo_{which}.npz", allow_pickle=False) as z:
        return build_character(**dict(z))


def _chain(character, pose):
    d = character.data
    r = axis_angle_to_matrix(pose)
    world, joint = np.empty_like(r), np.empty((character.nj, 3))
    for j, p in enumerate(d["parents"]):
        local = d["local_basis"][j] @ r[j]
        world[j] = local if p < 0 else world[p] @ local
        joint[j] = d["local_offset"][j] if p < 0 else joint[p] + world[p] @ d["local_offset"][j]
    return world, joint


def _joints(character, pose):
    return _chain(character, pose)[1]


def _between(a, b):
    """Smallest rotation taking the direction a to the direction b."""
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3)+k+k @ k/(1+c)


def _aim(character, pose, joint, child, direction):
    """Turn joint so that the bone towards child points along direction."""
    d = character.data
    world, position = _chain(character, pose)
    turn = _between(position[child]-position[joint], direction)
    parent = world[d["parents"][joint]]
    local = np.linalg.inv(d["local_basis"][joint]) @ parent.T @ turn @ world[joint]
    pose[joint] = matrix_to_axis_angle(local)


def reach(character, pose, side, wrist_goal, elbow_pole):
    """Analytic two-bone reach. side is 0 for the left arm, 1 for the right.

    elbow_pole is the world direction the elbow bends towards. It keeps the
    upper arm outside the torso, which numerical IK does not.
    """
    sem = character.data["semantic_map"]
    shoulder, elbow, wrist = sem[16+side], sem[18+side], sem[20+side]
    rest = _joints(character, np.zeros_like(pose))
    upper, lower = np.linalg.norm(rest[elbow]-rest[shoulder]), np.linalg.norm(rest[wrist]-rest[elbow])
    s = _joints(character, pose)[shoulder]
    span = wrist_goal-s
    dist = np.clip(np.linalg.norm(span), abs(upper-lower)+1e-4, upper+lower-1e-4)
    axis = span/np.linalg.norm(span)
    along = (upper*upper-lower*lower+dist*dist)/(2*dist)
    pole = np.asarray(elbow_pole, float)-(np.asarray(elbow_pole, float) @ axis)*axis
    pole /= np.linalg.norm(pole)
    e = s+along*axis+np.sqrt(max(upper*upper-along*along, 0))*pole
    _aim(character, pose, shoulder, elbow, e-s)
    _aim(character, pose, elbow, wrist, s+dist*axis-e)
    return pose


def belly_pose(source):
    """Both hands resting on the source stomach, elbows out and slightly back."""
    d = source.data
    v, sem = d["mesh_vertices"], d["semantic_map"]
    height = np.ptp(v[:, 1])
    y = d["rest_joints"][sem[0], 1]+.13*height
    belt = v[(d["mesh_part"] == 0) & (np.abs(v[:, 1]-y) < .04*height)]
    if not len(belt):
        raise ValueError("Source mesh has no belly surface")
    front = belt[:, 2].max()
    pose = np.zeros((source.nj, 3))
    for side, sign in ((0, 1), (1, -1)):
        reach(source, pose, side, np.array([sign*.082*height, y, front+.03]), (sign*.8, -.45, -.4))
    return pose


def procedural_motion(source, frames=120):
    """Move both hands from rest to the slim source belly, hold, and release.

    Endpoints are constructed from the source geometry with analytic reaching,
    without training or a recorded clip. Copying these local rotations to the
    heavier target creates a hand/forearm contact conflict.
    """
    if frames < 4:
        raise ValueError("At least four frames required")
    endpoint = belly_pose(source)
    t = np.linspace(0, 1, frames)
    a = np.clip(t/.3, 0, 1)
    a = a*a*(3-2*a)
    b = np.clip((1-t)/.25, 0, 1)
    b = b*b*(3-2*b)
    return np.minimum(a, b)[:, None, None]*endpoint


def hand_torso_depth(vertices, faces, parts, normals):
    """Full-vertex nearest-normal diagnostic, independent of pack sampling.

    Positive values are penetration in metres. This surface approximation is
    intended to compare the procedural example, not certify closed solids.
    """
    v, n = np.asarray(vertices), np.asarray(normals)
    torso = np.flatnonzero(parts == 0)
    arms = np.flatnonzero((parts >= 4) & (parts <= 7))
    b = v[torso]
    low, high = b.min(0), b.max(0)
    depths = []
    for start in range(0, len(arms), 128):
        q = v[arms[start:start+128]]
        near = np.argmin(((q[:, None]-b)**2).sum(-1), axis=1)
        disp = q-b[near]
        distance = np.linalg.norm(disp, axis=1)
        phi = (disp*n[torso[near]]).sum(-1)
        outside = np.linalg.norm(np.maximum(0, np.maximum(low-q, q-high)), axis=1)
        inside = (phi < 0) & ((distance < .012) | (phi < -.7*distance)) & (phi > -.30) & (outside <= .012)
        depths.extend(np.where(inside, -phi, 0))
    return np.asarray(depths)
