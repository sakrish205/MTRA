"""Source inventory: what was read, in which format, with which bytes.

Two jobs, both cheap and both load-bearing for reproducibility:

1. Hash and size every input, so a report can state exactly what it analysed.
2. Sniff the actual format instead of trusting the extension, because a binary
   STL named ``.stl`` and an ASCII STL named ``.stl`` need different handling and
   a mis-detection produces a plausible-looking but wrong mesh.

Nothing here writes to a source file. PRD FR-15 is enforced by construction: this
module opens files with ``"rb"`` only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from mtra.core.errors import InputError
from mtra.core.types import SourceFile

_ROBOT_EXTENSIONS = {
    ".ls": "fanuc",
    ".lst": "fanuc",
    ".mod": "abb",
    ".prg": "abb",
    ".tid": "durr",
    ".tip": "durr",
}


@dataclass(frozen=True)
class FormatGuess:
    format: str
    confident: bool
    detail: str


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_binary_stl(path: Path, size: int) -> bool:
    """A binary STL's declared triangle count must match its byte length.

    Checking the header word alone is unreliable: some exporters write "solid" at
    the start of a binary file.
    """
    if size < 84:
        return False
    with path.open("rb") as handle:
        header = handle.read(84)
    declared = int.from_bytes(header[80:84], "little")
    return 84 + 50 * declared == size


def detect_mesh_format(path: Path) -> FormatGuess:
    if not path.is_file():
        raise InputError(f"mesh file not found: {path}")
    size = path.stat().st_size
    suffix = path.suffix.lower()

    if suffix == ".obj":
        return FormatGuess("obj", True, "extension .obj")

    if suffix == ".stl":
        with path.open("rb") as handle:
            head = handle.read(512)
        if _is_binary_stl(path, size):
            return FormatGuess("stl-binary", True, "binary STL: triangle count matches file size")
        stripped = head.lstrip()
        if stripped[:5].lower() == b"solid" and b"facet" in head.lower():
            return FormatGuess("stl-ascii", True, "ASCII STL: solid header with facet records")
        if stripped[:5].lower() == b"solid":
            return FormatGuess(
                "stl-ascii",
                False,
                "starts with 'solid' but no facet record in the first 512 bytes; "
                "possibly a binary file with a 'solid' header",
            )
        return FormatGuess("stl-unknown", False, "no recognisable STL structure")

    return FormatGuess("unknown", False, f"unsupported mesh extension {suffix!r}")


def detect_robot_format(path: Path) -> str | None:
    return _ROBOT_EXTENSIONS.get(path.suffix.lower())


def inventory_file(path: Path, role: str) -> SourceFile:
    if not path.is_file():
        raise InputError(f"{role} file not found: {path}")
    if role == "mesh":
        guess = detect_mesh_format(path)
        detected = guess.format
    else:
        detected = detect_robot_format(path) or path.suffix.lower().lstrip(".") or "unknown"
    return SourceFile(
        path=str(path),
        role=role,
        sha256=sha256_of(path),
        size_bytes=path.stat().st_size,
        detected_format=detected,
    )
