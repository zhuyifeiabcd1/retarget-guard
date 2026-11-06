"""Run without SMPL files: two CC0 MPFB characters and procedural motion."""
import argparse
import json
from pathlib import Path
import numpy as np
from retarget_guard import RetargetGuard, vertex_normals
from retarget_guard.demo import demo_character, procedural_motion, hand_torso_depth


def run(frames=120, lookahead=5, output=None):
    source, target = demo_character("source"), demo_character("target")
    motion = procedural_motion(source, frames)
    d = target.data
    rows = []
    corrected = []
    with RetargetGuard(target, source, lookahead=lookahead) as guard:
        def consume(pose):
            frame = len(corrected)
            v0 = target.skin(motion[frame])
            v1 = target.skin(pose)
            n0, n1 = (vertex_normals(v, d["mesh_faces"]) for v in (v0, v1))
            v2 = guard.deform(v1, n1)
            n2 = vertex_normals(v2, d["mesh_faces"])
            rows.append([hand_torso_depth(v, d["mesh_faces"], d["mesh_part"], n).max()*1000
                         for v,n in ((v0,n0),(v1,n1),(v2,n2))])
            corrected.append(pose.copy())
        for pose in motion:
            result = guard.step(pose, source_pose=pose)
            if result is not None: consume(result)
        for result in guard.flush(): consume(result)
    depths = np.asarray(rows)
    hold = slice(int(.35*frames), int(.7*frames))
    report = dict(frames=frames,lookahead=lookahead,vertices=len(d["mesh_vertices"]),
        diagnostic="Full mesh forearm/hand-to-torso nearest-normal depth; no soft allowance",
        max_depth_mm=dict(zip(("copy","layer1","layer1_layer2"),depths.max(0).tolist())),
        hold_mean_depth_mm=dict(zip(("copy","layer1","layer1_layer2"),depths[hold].mean(0).tolist())))
    if not (depths[hold,0].mean() > 5 and depths[hold,2].mean() < depths[hold,0].mean()):
        raise RuntimeError("Procedural example did not demonstrate penetration reduction")
    if output is not None:
        output = Path(output); output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output,input_pose=motion,corrected_pose=np.asarray(corrected),depth_mm=depths)
    print(json.dumps(report,indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames",type=int,default=120)
    parser.add_argument("--lookahead",type=int,default=5,choices=range(6))
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    run(args.frames,args.lookahead,args.output)
