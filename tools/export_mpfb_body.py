"""Extract a generated MPFB Human body as generic builder inputs.

Input is the numeric mesh/rig export consumed by the research Rig.from_mpfb.
Only body geometry, weights and skeleton are copied. No MPFB code is needed.
"""
import argparse
from pathlib import Path
import numpy as np
from retarget_guard import build_character

BODY_NAMES = ("pelvis thigh_l thigh_r spine_01 calf_l calf_r spine_02 foot_l foot_r "
              "spine_03 ball_l ball_r neck_01 clavicle_l clavicle_r head upperarm_l "
              "upperarm_r lowerarm_l lowerarm_r hand_l hand_r").split()


def export(input_path, output_path):
    with np.load(input_path, allow_pickle=False) as z:
        vertices, faces, weights, helpers = (z["Human__"+key] for key in ("v","f","w","h"))
        valid = ~helpers
        bad = valid & (~np.isfinite(vertices).all(1) | ~np.isfinite(weights).all(1) |
                       (weights<0).any(1) | (weights.sum(1)<=0))
        if bad.sum() > .01*valid.sum():
            raise ValueError("Systematic invalid MPFB body rows")
        valid &= ~bad
        remap = np.full(len(vertices),-1)
        remap[valid] = np.arange(valid.sum())
        faces = remap[faces[valid[faces].all(1)]]
        ix = {name:i for i,name in enumerate(z["bone_names"].tolist())}
        c = np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)
        data = dict(vertices=vertices[valid]@c.T,faces=faces,skin_weights=weights[valid].astype(np.float32),
                    joint_parents=z["bone_parent"],rest_joints=z["bone_rest"][:,:3,3]@c.T,
                    rest_basis=c@z["bone_rest"][:,:3,:3],
                    semantic_map=np.array([ix[name] for name in BODY_NAMES]))
    build_character(**data)  # Verify the extracted character before saving.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open("wb") as f:
        np.savez_compressed(f,**data)
    print(f"Saved body with {valid.sum()} vertices; removed {helpers.sum()} helpers and {bad.sum()} bad rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    export(args.input,args.output)
