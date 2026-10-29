"""Install a Linux k-Wave BinaryPath wrapper for the separately built solver."""

import argparse
import os
import sys
import textwrap
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--directory", required=True, type=Path)
    p.add_argument("--batch-size", required=True, type=int)
    p.add_argument(
        "--library-dir",
        action="append",
        default=[],
        type=Path,
        help="Linux runtime library directory (repeatable); k-Wave clears LD_LIBRARY_PATH",
    )
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("--batch-size must be positive")
    binary = a.binary.resolve(strict=True)
    wrapper = Path(__file__).with_name("run_native_source_batch.py").resolve()
    a.directory.mkdir(parents=True, exist_ok=True)
    target = a.directory / "kspaceFirstOrder-CUDA"
    if target.exists():
        raise FileExistsError(target)
    # No shell interpolation. Python executable with spaces is launched by /usr/bin/env.
    target.write_text(
        textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import os
        import sys

        libraries = {os.pathsep.join(str(v.resolve(strict=True)) for v in a.library_dir)!r}
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            value for value in (libraries, os.environ.get("LD_LIBRARY_PATH", "")) if value
        )
        command = [
            {sys.executable!r},
            {str(wrapper)!r},
            "--native-binary", {str(binary)!r},
            "--batch-size", {str(a.batch_size)!r},
            *sys.argv[1:],
        ]
        os.execv(command[0], command)
        """),
        encoding="utf-8",
    )
    target.chmod(0o755)
    print(target)


if __name__ == "__main__":
    main()
