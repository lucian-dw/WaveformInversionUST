"""Install a Linux k-Wave BinaryPath wrapper for the separately built solver."""
import argparse
import os
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binary',required=True,type=Path)
    p.add_argument('--directory',required=True,type=Path)
    p.add_argument('--library-dir',action='append',default=[],type=Path,
                   help='Linux runtime library directory (repeatable); k-Wave clears LD_LIBRARY_PATH')
    a=p.parse_args()
    binary=a.binary.resolve(strict=True)
    wrapper=Path(__file__).with_name('run_native_source_batch.py').resolve()
    a.directory.mkdir(parents=True,exist_ok=True)
    target=a.directory/'kspaceFirstOrder-CUDA'
    if target.exists(): raise FileExistsError(target)
    # No shell interpolation. Python executable with spaces is launched by /usr/bin/env.
    target.write_text('#!/usr/bin/env python3\nimport os,sys\n'
        +f'os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(v for v in [{os.pathsep.join(str(v.resolve(strict=True)) for v in a.library_dir)!r}, os.environ.get("LD_LIBRARY_PATH", "")] if v)\n'
        +f'os.execv({sys.executable!r}, [{sys.executable!r}, {str(wrapper)!r}, '
        +f'"--native-binary", {str(binary)!r}, "--batch-size", "128", *sys.argv[1:]])\n')
    target.chmod(0o755)
    print(target)

if __name__=='__main__': main()
