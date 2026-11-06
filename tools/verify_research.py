"""Local parity audit. The research checkout and licensed data stay external.

Run with --research-root PATH. Never invokes an original build or writes there.
Only numeric differences and environment metadata are saved, never motion or
SMPL vertex arrays. Use the reference interpreter requested for your audit.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from retarget_guard import RetargetGuard, from_smplh
from retarget_guard._build import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-root",type=Path,required=True)
    args = ap.parse_args()
    root = args.research_root.resolve()
    out = Path(__file__).resolve().parents[1]/"verification"
    out.mkdir(exist_ok=True)
    # Build the original source in isolation, using exactly its original flags.
    spec = importlib.util.spec_from_file_location("reference_guard",root/"guard/guard.py")
    ref = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)
    ref.BUILD = out.parent/".cache/reference_build"
    ref.BUILD.mkdir(parents=True,exist_ok=True)
    compiler = os.environ.get("CC","clang")
    subprocess.run([compiler,*ref.FLAGS,str(root/"guard/guard_core.c"),"-lm","-o",str(ref.BUILD/"guard_core.so")],check=True)
    # The package must use identical compiler and numerical flags for bit parity.
    os.environ["CC"] = compiler
    os.environ["RETARGET_GUARD_MARCH"] = "x86-64-v3"
    os.environ.pop("CFLAGS",None)
    build("guard_core",force=True)
    sys.path[:0] = [str(root/"s2/code"),str(root/"s0/code")]
    import benchmark_mesh7 as m7
    conf = json.loads((root/"guard/settings.json").read_text())["settings"]
    parity = []
    for clip,char in ((555,"tall_heavy"),(272,"tall_heavy"),(406,"short_heavy")):
        src = m7.V.problem("smplh","source",clip)
        src.source()
        p,sr = m7.get_problem("smplh",char,clip,src)
        rotations = p.R[0].numpy()
        target_path = root/"s2/results/stream7/packs/smplh/full"/(char+".npz")
        source_path = target_path.with_name(f"source_{clip}.npz")
        # Require existing packs so the benchmark never prepares protected data.
        if not target_path.exists() or not source_path.exists():
            raise FileNotFoundError("Pre-existing reference packs are required")
        for horizon in (0,5):
            target = m7.pack("smplh",char,clip,"full",ref)
            source = m7.pack("smplh","source",clip,"full",ref)
            baseline = ref.run(target,source,rotations,sr,horizon,ref.Settings(**conf))[0]
            target.close(); source.close()
            values = []
            with RetargetGuard(target_path,source_path,lookahead=horizon) as guard:
                for a,b in zip(rotations,sr):
                    pose = guard.step(a,source_pose=b)
                    if pose is not None: values.append(guard.last_delta.copy())
                for pose in guard.flush(): values.append(guard.last_delta.copy())
            actual = np.asarray(values)
            row = dict(clip=clip,character=char,lookahead=horizon,frames=len(actual),
                       max_abs=float(np.max(np.abs(actual-baseline))),
                       bit_identical=actual.tobytes()==baseline.tobytes(),
                       nonzero_frames=int(np.any(baseline!=0,axis=(1,2)).sum()))
            parity.append(row)
            print("PARITY",row,flush=True)
    pack_rows = {}
    with np.load(root/"s0/data/shapes_smplh.npz",allow_pickle=False) as shapes:
        for shape in ("mid","tall_heavy"):
            new = from_smplh(root/"s0/data/smplh_lite.npz",shapes[shape]).data
            rows = {}
            path = root/"s2/results/stream7/packs/smplh/full"/(shape+".npz")
            with np.load(path,allow_pickle=False) as old:
                for field in old.files:
                    a,b = new[field],old[field]
                    rows[field] = dict(shape=list(a.shape),same_shape=a.shape==b.shape,
                        dtype=str(a.dtype),reference_dtype=str(b.dtype),
                        max_abs=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if a.shape==b.shape else None,
                        bit_identical=a.dtype==b.dtype and a.tobytes()==b.tobytes())
            pack_rows[shape] = rows
            print("PACK",shape,json.dumps(rows),flush=True)
    result = dict(compiler=subprocess.check_output([compiler,"--version"],text=True).splitlines()[0],
                  flags=ref.FLAGS,clips=parity,packs=pack_rows,
                  settings_verbatim=(root/"guard/settings.json").read_bytes()==
                    (Path(__file__).resolve().parents[1]/"src/retarget_guard/settings.json").read_bytes())
    (out/"parity.json").write_text(json.dumps(result,indent=2)+"\n")
    if any(row["max_abs"]!=0 or not row["bit_identical"] for row in parity):
        raise AssertionError("Wrapper delta parity failed")


if __name__=="__main__": main()
