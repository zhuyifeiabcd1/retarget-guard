"""Lazy C99 compilation into a user-writable cache; no numerical code changes."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

SOURCES = Path(__file__).resolve().parent / "c"


def build(name, *, force=False):
    if name not in ("guard_core", "guard_core_cloth", "deform_core"):
        raise ValueError("Unknown C library")
    cache = Path(os.environ.get("RETARGET_GUARD_CACHE", str(
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "retarget_guard")))
    cache.mkdir(parents=True, exist_ok=True)
    source = SOURCES / (name + ".c")
    dest = cache / (name + ".so")
    if not force and dest.exists() and dest.stat().st_mtime >= source.stat().st_mtime:
        return dest
    flags = ["-std=c99", "-O3", "-fno-math-errno"]
    if name != "deform_core":
        flags.append("-ffp-contract=fast")
    march = os.environ.get("RETARGET_GUARD_MARCH", "")
    if march:
        flags.append("-march=" + march)
    flags += ["-fPIC", "-shared"]
    flags += shlex.split(os.environ.get("CFLAGS", ""))
    compiler = shlex.split(os.environ.get("CC", "cc"))
    # Unique build file and atomic rename support concurrent first use.
    fd, temporary = tempfile.mkstemp(prefix=name + "-", suffix=".so", dir=cache)
    os.close(fd)
    try:
        subprocess.run([*compiler, *flags, str(source), "-lm", "-o", temporary], check=True)
        os.replace(temporary, dest)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return dest


def library_path(name):
    return build(name)


if __name__ == "__main__":
    for library in ("guard_core", "guard_core_cloth", "deform_core"):
        print(build(library, force=True))
