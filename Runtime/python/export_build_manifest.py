"""Write verified release identity into an already assembled source archive."""

import argparse

from wust_io import atomic_json

from provenance import identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    info = identity()
    if (
        info["dirty"] is not False
        or not info["git_sha"]
        or not info["source_manifest_verified"]
    ):
        raise ValueError("Build identity requires a clean verified Git checkout")
    atomic_json(
        args.output,
        {
            "schema": "wust.build",
            "schema_version": 1,
            "git_sha": info["git_sha"],
            "runtime_version": info["runtime_version"],
            "source_manifest_sha256": info["source_manifest_sha256"],
        },
    )


if __name__ == "__main__":
    main()
