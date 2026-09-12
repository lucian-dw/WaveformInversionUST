"""Thin, dependency-free MATLAB launcher. Geometry/arrays remain caller-owned."""
import argparse
import json
from pathlib import Path
import subprocess
import time

VERSION='0.1.0'

def run(request_path, matlab='matlab', timeout=21600):
    request_path=Path(request_path).resolve(strict=True)
    req=json.loads(request_path.read_text(encoding='utf-8'))
    if req.get('schema')!='wfi.request.v1': raise ValueError('Unsupported schema')
    if req.get('operation') not in {'simulate','prepare','reconstruct'}: raise ValueError('Unsupported operation')
    for key in ('input_mat','output_mat'):
        if not Path(req[key]).is_absolute(): raise ValueError(key+' must be absolute')
    if not Path(req['input_mat']).is_file(): raise FileNotFoundError(req['input_mat'])
    if Path(req['output_mat']).exists(): raise FileExistsError(req['output_mat'])
    matlab_dir=Path(__file__).resolve().parents[1]/'matlab'
    quote=lambda v: "'"+str(v).replace("'","''")+"'"
    expression='addpath('+quote(matlab_dir)+');'
    if req.get('kwave_toolbox_path'):
        toolbox=Path(req['kwave_toolbox_path']).resolve(strict=True)
        expression+='addpath('+quote(toolbox)+');'
    expression+='wfi_run('+quote(request_path)+');'
    started=time.perf_counter()
    # No shell=True; stdout streams live; MATLAB -batch gives a nonzero exit on errors.
    subprocess.run([str(matlab),'-batch',expression],check=True,timeout=timeout)
    if not Path(req['output_mat']).is_file(): raise RuntimeError('MATLAB returned without result')
    return {'version':VERSION,'output_mat':req['output_mat'],'process_wall_seconds':time.perf_counter()-started}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('request');p.add_argument('--matlab',default='matlab');p.add_argument('--timeout',type=float,default=21600)
    a=p.parse_args();print(json.dumps(run(a.request,a.matlab,a.timeout),indent=2))

if __name__=='__main__': main()
