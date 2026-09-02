"""Functions that are used in tests."""

import hashlib
import os
import shutil
import socket
import urllib.request
from pathlib import Path


def on_cisl_machine():
    """Return True if the current machine is a CISL machine, False otherwise."""
    fqdn = socket.getfqdn()
    return "hpc.ucar.edu" in fqdn


# ---------------------------------------------------------------------------
# CESM input data
# ---------------------------------------------------------------------------

CESM_INPUTDATA_URL = "https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata"

# Files fetched on demand by the test suite, keyed by their path relative to
# the CESM inputdata root. Checksums guard against truncated or corrupted
# downloads; regenerate with `shasum -a 256` if a file is ever revised.
INPUTDATA_FILES = {
    "share/meshes/tx2_3v2_230415_ESMFmesh.nc": (
        "0b19fbd69dcc9bee20125aec9b3b899d0f7ac40f37dafc7a6e69bd1244861271"
    ),
    "share/meshes/gx1v7_151008_ESMFmesh.nc": (
        "b1b892dfa5da00447c35b58a5bc3a35913e6eb1330b48ae46c2e8bcb88a3c641"
    ),
    "share/meshes/rx1_nomask_181022_ESMFmesh.nc": (
        "e67e140e6df410d3ca2b2a574d82959433a1bfabad464e71240370f83f0c41d4"
    ),
}

GLADE_INPUTDATA = Path("/glade/campaign/cesm/cesmdata/inputdata")


def inputdata_cache_dir():
    """Directory that downloaded CESM input data is cached in.

    Override with MOM6_FORGE_TEST_DATA (useful for pointing CI at a cached
    path, or for sharing one download across checkouts).
    """
    env = os.environ.get("MOM6_FORGE_TEST_DATA")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "mom6_forge" / "inputdata"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_inputdata(relpath):
    """Return a local path to a CESM input data file, downloading if needed.

    Resolution order, so that nobody downloads what they already have:

      1. ``$CESMDATAROOT`` or ``$MOM6_FORGE_INPUTDATA``, if either points at a
         tree containing the file.
      2. GLADE, when running on a CISL machine.
      3. The local cache, downloading from the public CESM input data server
         on a miss.

    Raises
    ------
    RuntimeError
        If the file cannot be obtained. Tests are expected to fail rather than
        skip in that case, so that missing coverage is never silent.
    """
    relpath = str(relpath)
    if relpath not in INPUTDATA_FILES:
        raise KeyError(
            f"{relpath!r} is not a known test input file; add it to INPUTDATA_FILES"
        )
    expected = INPUTDATA_FILES[relpath]

    # 1. an existing inputdata tree
    for var in ("CESMDATAROOT", "MOM6_FORGE_INPUTDATA"):
        root = os.environ.get(var)
        if root:
            candidate = Path(root) / relpath
            if candidate.exists():
                return candidate

    # 2. GLADE
    if on_cisl_machine():
        candidate = GLADE_INPUTDATA / relpath
        if candidate.exists():
            return candidate

    # 3. local cache, downloading on a miss
    dest = inputdata_cache_dir() / relpath
    if dest.exists():
        if _sha256(dest) == expected:
            return dest
        dest.unlink()  # corrupt or stale; re-fetch

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{CESM_INPUTDATA_URL}/{relpath}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as exc:  # network, DNS, HTTP error, ...
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}.\n"
            f"Set CESMDATAROOT or MOM6_FORGE_INPUTDATA to an existing inputdata "
            f"tree, or MOM6_FORGE_TEST_DATA to a writable cache directory."
        ) from exc

    actual = _sha256(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {url}\n  expected {expected}\n  got      {actual}"
        )
    tmp.rename(dest)
    return dest
