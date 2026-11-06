import sys
import numpy as np
import pytest
from retarget_guard import Character, RetargetGuard, build_character, frozen_settings, vertex_normals
from retarget_guard._native import Pack, Guard
from retarget_guard.demo import procedural_motion
from retarget_guard.rotations import axis_angle_to_matrix, matrix_to_axis_angle


def test_roundtrip_character_and_bind_skinning(character, tmp_path):
    path = character.save(tmp_path/"character.npz")
    copy = Character.load(path)
    for key in character.data:
        assert np.array_equal(copy.data[key],character.data[key])
    rest = copy.skin(np.zeros((copy.nj,3)))
    # Exported float32 bind bases accumulate sub-micrometre FK roundoff.
    np.testing.assert_allclose(rest,copy.data["mesh_vertices"],atol=1e-6)


@pytest.mark.parametrize("lookahead",range(6))
def test_stream_matches_explicit_windows(character, lookahead):
    motion = procedural_motion(character,12)
    r = axis_angle_to_matrix(motion)
    target,source = Pack(character),Pack(character)
    native = Guard(target,source,frozen_settings())
    expected,poses = [],[]
    try:
        for t in range(len(r)):
            delta,_ = native.step(r[t:t+lookahead+1],r[t:t+lookahead+1])
            expected.append(delta.copy()); poses.append(native.pose.copy())
    finally:
        native.close();target.close();source.close()
    actual,outputs = [],[]
    with RetargetGuard(character,character,lookahead) as stream:
        for i,frame in enumerate(r):
            buffer = frame.copy()
            out = stream.step(buffer)
            buffer[:] = 0  # Buffered inputs must not alias caller memory.
            if i<lookahead: assert out is None
            if out is not None:
                outputs.append(out);actual.append(stream.last_delta.copy())
        for out in stream.flush():
            outputs.append(out);actual.append(stream.last_delta.copy())
        assert list(stream.flush())==[]
        with pytest.raises(RuntimeError,match="flushed"):stream.step(r[0])
    assert np.asarray(actual).tobytes()==np.asarray(expected).tobytes()
    assert np.asarray(outputs).tobytes()==np.asarray(poses).tobytes()


@pytest.mark.parametrize("axis",[[0,0,0],[1e-10,-2e-10,3e-10],[.3,-.5,.8],[np.pi,0,0],[0,np.pi-1e-8,0]])
def test_rotation_roundtrip(axis):
    r = axis_angle_to_matrix(np.array(axis))
    np.testing.assert_allclose(axis_angle_to_matrix(matrix_to_axis_angle(r)),r,atol=1e-8)


def test_axis_angle_and_matrix_stream_agree(character):
    motion = procedural_motion(character,15)
    with RetargetGuard(character,character) as a, RetargetGuard(character,character) as b:
        for frame in motion:
            aa,rm = a.step(frame),b.step(axis_angle_to_matrix(frame))
            np.testing.assert_allclose(axis_angle_to_matrix(aa),rm,atol=1e-12)
            assert a.last_delta.tobytes()==b.last_delta.tobytes()


def test_short_clip_flush(character):
    with RetargetGuard(character,character,5) as stream:
        for _ in range(2):assert stream.step(np.zeros((character.nj,3))) is None
        assert len(list(stream.flush()))==2


def test_c_return_failure_raises_and_invalidates_stream(character,monkeypatch):
    with RetargetGuard(character,character) as stream:
        monkeypatch.setattr(stream._guard.lib,"guard_step",lambda *args:-30)
        with pytest.raises(RuntimeError,match="C guard step failed: -30"):
            stream.step(np.zeros((character.nj,3)))
        with pytest.raises(RuntimeError,match="Stream failed"):
            stream.step(np.zeros((character.nj,3)))


def test_deformer_stateless_and_c_failure(character,monkeypatch):
    with RetargetGuard(character,character) as stream:
        pose = stream.step(procedural_motion(character,12)[5])
        v = character.skin(pose)
        n = vertex_normals(v,character.data["mesh_faces"])
        a = stream.deform(v,n)
        rest = character.data["mesh_vertices"]
        stream.deform(rest,vertex_normals(rest,character.data["mesh_faces"]))
        b = stream.deform(v,n)
        assert a.tobytes()==b.tobytes()
        assert np.isfinite(a).all()
        monkeypatch.setattr(stream._deformer.lib,"deform_step",lambda *args:-2)
        with pytest.raises(RuntimeError,match="C deformation failed: -2"):stream.deform(v,n)


def test_cloth_kappa_one_matches_body(character):
    data = dict(character.data)
    data["cloth_offset"] = np.full(len(data["vertices"]),.025)
    cloth = Character(data)
    with RetargetGuard(character,character) as a, RetargetGuard(cloth,character,kappa=1) as b:
        for frame in procedural_motion(character,15):
            a.step(frame);b.step(frame)
            assert a.last_delta.tobytes()==b.last_delta.tobytes()


def test_cloth_hard_surface_runs(character):
    data = dict(character.data)
    data["cloth_offset"] = np.full(len(data["vertices"]),.025)
    with RetargetGuard(Character(data),character,kappa=0) as stream:
        result = stream.step(procedural_motion(character,12)[5])
        assert np.isfinite(result).all()


def test_invalid_input_and_closed_stream(character):
    stream = RetargetGuard(character,character)
    for bad in (np.zeros((3,3)),np.full((character.nj,3),np.nan),np.zeros((character.nj,3,3))):
        with pytest.raises(ValueError):stream.step(bad)
    stream.close();stream.close()
    with pytest.raises(RuntimeError,match="closed"):stream.step(np.zeros((character.nj,3)))


@pytest.mark.parametrize("kwargs",[{"lookahead":6},{"lookahead":1.5},{"fps":0},{"kappa":-1}])
def test_invalid_configuration(character,kwargs):
    with pytest.raises(ValueError):RetargetGuard(character,character,**kwargs)


def test_explicit_source_required_for_different_bind(character):
    data = dict(character.data)
    data["local_basis"] = data["local_basis"].copy()
    data["local_basis"][0] = axis_angle_to_matrix(np.array([.1,0,0]))
    with RetargetGuard(character,Character(data)) as stream:
        with pytest.raises(ValueError,match="source_pose"):stream.step(np.zeros((character.nj,3)))


def test_frozen_configuration():
    config=frozen_settings()
    assert config.w_vel==800000 and config.w_acc==3600000 and config.urg_smooth==16
    assert "torch" not in sys.modules
