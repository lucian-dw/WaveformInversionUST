"""Source identity without inventing Git facts for archives."""

import json
import subprocess
from pathlib import Path

from wust_io import sha256

ROOT = Path(__file__).resolve().parents[2]


def identity():
    source = ROOT / "provenance/source-manifest.json"
    report = {
        "runtime_name": "WUST",
        "runtime_version": (ROOT / "Runtime/VERSION").read_text().strip(),
        "upstream": json.loads((ROOT / "provenance/upstream.json").read_text()),
        "source_manifest_sha256": sha256(source),
        "git_sha": None,
        "dirty": None,
        "identity_source": "source_manifest",
    }
    hashes = json.loads(source.read_text())["sha256"]
    report["source_manifest_verified"] = all(
        (ROOT / p).is_file() and sha256(ROOT / p) == h for p, h in hashes.items()
    )
    if (ROOT / ".git").exists():
        try:

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(ROOT), *args],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).strip()

            if Path(git("rev-parse", "--show-toplevel")).resolve() == ROOT.resolve():
                report.update(
                    git_sha=git("rev-parse", "HEAD"),
                    dirty=bool(git("status", "--porcelain")),
                    identity_source="git",
                )
        except (OSError, subprocess.SubprocessError):
            report["git_error"] = "Git identity unavailable"
    build = ROOT / "provenance/build-manifest.json"
    if report["git_sha"] is None and build.exists():
        from contracts import schema

        data = json.loads(build.read_text())
        schema(data, "wust.build")
        if data["source_manifest_sha256"] != report["source_manifest_sha256"]:
            raise ValueError("Release build/source manifest mismatch")
        report.update(build_manifest=data, identity_source="build_manifest")
    return report
