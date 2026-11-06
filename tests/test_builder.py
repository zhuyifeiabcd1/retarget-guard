import numpy as np
import pytest
from retarget_guard import build_character, RetargetGuard, SEMANTICS


def test_generic_semantics_and_extra_bone_inheritance(raw_mesh):
    d = dict(raw_mesh)
    sem = d.pop("semantic_map")
    character = build_character(**d,semantic_map=dict(zip(SEMANTICS,sem)))
    np.testing.assert_array_equal(character.data["semantic_map"],sem)
    assert len(character.data["vertices"])>1000
    assert (character.data["mesh_part"]==6).sum()>100
    assert np.isfinite(character.data["mesh_soft"]).all()


def test_isolated_bad_vertex_is_skipped_and_counted(raw_mesh):
    d = {k:v.copy() for k,v in raw_mesh.items()}
    # Add an unreferenced bad vertex, which must not poison the character.
    d["vertices"] = np.r_[d["vertices"],[[np.nan,0,0]]]
    d["skin_weights"] = np.r_[d["skin_weights"],np.zeros((1,len(d["joint_parents"])))]
    with pytest.warns(RuntimeWarning,match="Skipped 1 bad vertices"):
        character = build_character(**d)
    assert int(character.data["dropped_vertices"])==1
    assert int(character.data["dropped_faces"])==0


def test_systematic_bad_weights_stop(raw_mesh):
    d = dict(raw_mesh); d["skin_weights"]=d["skin_weights"].copy()
    d["skin_weights"][:len(d["vertices"])//10]=0
    with pytest.raises(ValueError,match="Systematic"):build_character(**d)


def test_bad_face_is_skipped_and_counted(raw_mesh):
    d = dict(raw_mesh);d["faces"]=np.r_[d["faces"],[[-1,0,1]]]
    with pytest.warns(RuntimeWarning,match="1 faces"):
        character=build_character(**d)
    assert int(character.data["dropped_faces"])==1


def test_bad_hierarchy_and_semantics_stop(raw_mesh):
    d = dict(raw_mesh);d["joint_parents"]=d["joint_parents"].copy()
    d["joint_parents"][1]=1
    with pytest.raises(ValueError,match="hierarchy"):build_character(**d)
    d = dict(raw_mesh);d["semantic_map"]=np.zeros(22,dtype=int)
    with pytest.raises(ValueError,match="distinct"):build_character(**d)


def test_different_joint_counts_end_to_end(raw_mesh,character):
    d = dict(raw_mesh)
    wrist = int(d["semantic_map"][20])
    d["joint_parents"] = np.r_[d["joint_parents"],wrist]
    d["rest_joints"] = np.r_[d["rest_joints"],d["rest_joints"][[wrist]]]
    d["rest_basis"] = np.concatenate([d["rest_basis"],d["rest_basis"][[wrist]]])
    d["skin_weights"] = np.c_[d["skin_weights"],np.zeros(len(d["vertices"]))]
    extended = build_character(**d)
    with RetargetGuard(extended,character) as guard:
        pose = np.zeros((extended.nj,3))
        pose[-1] = [.1,.2,.3]
        with pytest.raises(ValueError,match="source_pose"):guard.step(pose)
        result = guard.step(pose,source_pose=np.zeros((character.nj,3)))
        assert result.shape==pose.shape
        np.testing.assert_array_equal(result[-1],pose[-1])
        vertices = extended.skin(result)
        from retarget_guard import vertex_normals
        final = guard.deform(vertices,vertex_normals(vertices,extended.data["mesh_faces"]))
        assert np.isfinite(final).all()
