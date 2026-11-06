import os
from pathlib import Path
import pytest

# Keep compiler output and test scratch within the release workspace.
os.environ.setdefault("RETARGET_GUARD_CACHE", str(Path(__file__).resolve().parents[1]/".cache/native"))


@pytest.fixture(scope="session")
def character():
    from retarget_guard.demo import demo_character
    return demo_character("target")


@pytest.fixture(scope="session")
def raw_mesh():
    import numpy as np
    import retarget_guard
    path = Path(retarget_guard.__file__).with_name("data")/"demo_target.npz"
    with np.load(path,allow_pickle=False) as z:
        return dict(z)
