"""Provenance helpers shared by the stale-regeneration producers/consumers
(review round 49): full hashes, declared partial fingerprints of very large
read-only inputs, and directory digests."""
import glob
import hashlib
import os

PARTIAL_BYTES = 16 << 20


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def partial_fp(p, nbytes=PARTIAL_BYTES):
    """Size, mtime and sha256 of the first `nbytes` -- a declared PARTIAL
    fingerprint for multi-GB raw cutouts, not a full content hash."""
    st = os.stat(p)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(nbytes))
    return dict(size=st.st_size, mtime_ns=st.st_mtime_ns, sha256_head=h.hexdigest(), head_bytes=nbytes)


def stat_fp(p):
    st = os.stat(p)
    return dict(size=st.st_size, mtime_ns=st.st_mtime_ns)


def dir_digest(d, pattern="*.npz"):
    """(count, sha256 over the sorted 'name:sha256' lines) of a directory's files."""
    files = sorted(glob.glob(os.path.join(d, pattern)))
    h = hashlib.sha256()
    for p in files:
        h.update(f"{os.path.basename(p)}:{sha(p)}\n".encode())
    return dict(dir=d, pattern=pattern, count=len(files), digest=h.hexdigest())
