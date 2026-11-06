# Verification report

All commands used the allocated CPU cores 24–25 with one numerical thread and the requested scratch directory. Research files were read only. The original guard build function was never called; the reference C source was compiled into this release's private cache. No push or remote repository operation was performed.

## Wrapper parity

The audit used the requested `rdf_venv/bin/python` interpreter and loaded problems through `benchmark_mesh7`, following `guard/exp_fast/same.py`. The original source-body observations were prepared as in that script. The reference and release used the same existing full-density packs, h13 settings and compiler options. Both zero-lookahead and five-frame lookahead were checked, including the final shortened windows from `flush()`.

Compiler: `clang version 18.1.8 (Red Hat 18.1.8-1.module+el8.10.0+703+ec7b33ba)`. Flags: `-std=c99 -O3 -fno-math-errno -ffp-contract=fast -march=x86-64-v3 -fPIC -shared`.

| Clip | Character | Lookahead | Frames | Max absolute delta difference (radians) | Identical bits |
|---:|---|---:|---:|---:|---|
| 555 | tall_heavy | 0 | 300 | 0.0 | True |
| 555 | tall_heavy | 5 | 300 | 0.0 | True |
| 272 | tall_heavy | 0 | 300 | 0.0 | True |
| 272 | tall_heavy | 5 | 300 | 0.0 | True |
| 406 | short_heavy | 0 | 300 | 0.0 | True |
| 406 | short_heavy | 5 | 300 | 0.0 | True |

All six comparisons are bit-identical, for 1,800 output frames in total. This is a numerical output comparison, not binary-file identity or a claim across different compilers/flags. All three C sources retain the original executable code; only comments were changed. The portable install default omits `-march`, so explicitly request the flags above when reproducing this parity.

`settings.json` matches the original byte for byte: **True**.

## Rebuilt pack parity

`from_smplh` calls the new NumPy `build_character` on the user-owned model and each shape. Sampling is performed on the reference template in float64, as in the original `BodyModel._build_sets`; shape evaluation, normals, skeleton and soft budgets use float32. Existing comparison packs were read from the original full-density SMPL-H pack directory. Neither the model nor any resulting SMPL mesh/pack was saved to the release.

Both packs contain 800 query vertices and 3,821 collider vertices. Every original field has the same shape and dtype. The following table gives max absolute element differences; positions, offsets and thickness are in metres, while normals and rotations are dimensionless.

| Field | mid max abs | tall_heavy max abs | mid identical bits | tall_heavy identical bits |
|---|---:|---:|---|---|
| `local_basis` | 0 | 0 | True | True |
| `local_offset` | 4.172325134e-07 | 5.36441803e-07 | False | False |
| `rest_basis` | 0 | 0 | True | True |
| `rest_joints` | 3.576278687e-07 | 2.980232239e-07 | False | False |
| `parents` | 0 | 0 | True | True |
| `semantic_map` | 0 | 0 | True | True |
| `vertices` | 1.192092896e-07 | 1.192092896e-07 | False | False |
| `normals` | 1.11758709e-06 | 7.033348083e-06 | False | False |
| `wi` | 0 | 0 | True | True |
| `weights` | 0 | 0 | True | True |
| `soft` | 1.303851604e-08 | 1.490116119e-08 | False | False |
| `part` | 0 | 0 | True | True |
| `query` | 0 | 0 | True | True |
| `vertex_ids` | 0 | 0 | True | True |
| `q_indices` | 0 | 0 | True | True |
| `pairs` | 0 | 0 | True | True |
| `max_depth` | 0 | 0 | True | True |

Pack floating-point fields are **not bit-identical** to Torch preprocessing. Geometry differs by at most 0.1193 micrometres, rest joints by 0.3577 micrometres, local offsets by 0.5365 micrometres, and normals by 7.0334e-6. The discrete sets, skin weights and every other original field are exact. The wrapper bit-parity result above uses the same existing packs on both sides; it must not be interpreted as bit parity after rebuilding packs.

Pack generation trace: `s0lib.BodyModel._build_sets` derives parts from summed joint weights, chooses high-purity query/collider vertices by farthest-point sampling and retains top-four normalized weights. `Problem.set_lambda` evaluates shape, normals and segment-distance soft thickness. `benchmark_mesh5.write_pack` delegates the full-density pack to `benchmark_mesh4.write_pack`, which concatenates queries/colliders and adds skeleton, semantics, pair categories and maximum depths. `benchmark_mesh7.pack` loads those files. MPFB follows `Rig.from_mpfb -> MeshLayer -> RigBody -> benchmark_mesh4.write_pack`; the release derives the equivalent attributes directly from mesh/weights/skeleton.

## Tests

Clang run (actual output):

```text
...............................                                          [100%]
31 passed in 4.68s
```

Default `cc` run, with the portable architecture default (actual output):

```text
...............................                                          [100%]
31 passed in 11.03s
```

Tests cover every lookahead 0–5 against explicit native windows, short-clip flushing, input-buffer reuse, axis-angle/matrix equivalence and pi rotations, C error propagation, stateless deformation, kappa=1 parity and kappa=0 execution, invalid configurations, bad-row counting/systematic failure, serialization, and a target with a different joint count. The latter builds an extra-joint target only from mesh, weights and skeleton and executes both layers.

The tests ran in a release-local venv with the existing NumPy available. Network package installation was unavailable, so pytest and its pure-Python support files were copied from an existing local environment into that venv. Shared environments were not modified.

Installed-wheel smoke test (actual output):

```text
Installed wheel import: site-packages
Blocked Torch, SciPy and research modules: Layer 1, cloth and Layer 2 passed
```

The wheel was installed without dependencies into that venv. The smoke process blocked imports of Torch, SciPy and the research modules, then constructed an MPFB character and ran Layer 1, the cloth kernel and Layer 2 from the installed package.

## Demo and asset license

The two demo bodies are MPFB-generated slim and heavy bodies, each with 13,380 vertices and 53 joints. Only the Human body, weights and skeleton were retained; helpers, garments, shoes and textures were excluded. Both fresh packs contain 4,974 sampled vertices. No bad body rows were found. The example generates its motion with deterministic analytic reaching; it does not use a recorded motion, SMPL model or external motion model.

MPFB asset/output licensing was checked in the local `LICENSE.md` sections C/D and against the [official MPFB license](https://github.com/makehumancommunity/mpfb2/blob/master/LICENSE.md). Base geometry, targets and rigs are CC0 1.0; exported/generated graphical data is covered by the output provision. MPFB source code has a separate GPLv3 license. This package includes numeric assets, the CC0 dedication and provenance; no MPFB program logic is copied.

The demo used default `cc` without `-march` and 120 frames at 30 fps with lookahead 5. These are full-mesh forearm/hand-to-torso nearest-normal depth diagnostics with no soft allowance; they use every relevant vertex rather than the solver pack samples.

| Diagnostic (mm) | Copy | Layer 1 | Layer 1 + Layer 2 |
|---|---:|---:|---:|
| Maximum over clip | 36.376917 | 7.303863 | 1.726580 |
| Mean frame maximum over hold | 36.376917 | 3.540146 | 0.430375 |

The example reduces the measured overlap but retains a residual; it does not establish collision-free motion or accuracy for other characters. This is a geometric surface diagnostic, not a closed-solid intersection proof.

## Validation limits

No general real-time motion-model integration or quality benchmark was run. No real garment pack or separate garment deformation was validated; cloth checks cover synthetic offsets, kappa=1 bit parity and kappa=0 execution. SMPL pose blendshapes, sparse/pickled model files, nonhumanoid/missing-region meshes, Windows and macOS were not validated. The numeric demo uses generated asset exports already present locally; Blender generation was not rerun.

## Distribution contents

The wheel and source archive were inspected. Both contain only the two documented MPFB NPZ assets, alongside C/Python sources and frozen settings. No SMPL-family geometry, recorded motion, compiled shared libraries, bytecode or caches are distributed. Text files contain no private machine paths or Chinese code comments. See `archive_audit.txt`. The wheel and source archive were built offline; the wheel was installed and exercised from site-packages.
