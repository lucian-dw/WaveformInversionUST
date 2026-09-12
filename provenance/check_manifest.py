"""Check published source integrity; --write refreshes hashes after review."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance/source-manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    paths = [ROOT / "README.md", ROOT / "LICENSE.txt", ROOT / ".gitignore"]
    for directory in ("Runtime", "docs", "provenance", "reference"):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or path == MANIFEST:
                continue
            if "__pycache__" in path.parts or "artifacts" in path.parts:
                continue
            if path.suffix in {".pyc", ".o", ".log"} or path.suffix.startswith(".mex"):
                continue
            if path.name == "kspaceFirstOrder-CUDA" or path.name == ".DS_Store":
                continue
            paths.append(path)
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def check():
    expected = json.loads(MANIFEST.read_text())["sha256"]
    actual = inventory()
    changed = sorted(
        k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)
    )
    if changed:
        raise ValueError("Manifest mismatch: " + ", ".join(changed))
    source_map = json.loads((ROOT / "provenance/source-map.json").read_text())
    for entry in source_map["files"]:
        if (
            entry["byte_preserved"]
            and digest(ROOT / entry["destination"]) != entry["original_sha256"]
        ):
            raise ValueError("Historical/vendor file changed: " + entry["destination"])
    return len(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        MANIFEST.write_text(
            json.dumps(
                {
                    "schema": "wust.source-manifest",
                    "version_file": "Runtime/VERSION",
                    "sha256": inventory(),
                },
                indent=2,
            )
            + "\n"
        )
    print(f"Manifest verified: {check()} files")


if __name__ == "__main__":
    main()
