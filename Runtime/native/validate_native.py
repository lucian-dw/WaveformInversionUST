"""A100 release gate: identical HDF5 input, serial vs independent-TX batching.

Requires a SINGLE-TX, full-128-sensor k-Wave HDF5 input (SaveToDisk).
No patient data required. Use a tiny asymmetric numerical medium first.
Output files are retained. Device and fresh output directory are mandatory.
The serial reference uses the same binary with native mode disabled.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import h5py
import numpy as np
from run_native_source_batch import expand_input

def pressure(path,nrx):
    with h5py.File(path) as h:a=np.asarray(h['p'][...]).squeeze()
    if a.ndim!=2:raise ValueError('Expected pressure matrix')
    if a.shape[1]==nrx:return a
    if a.shape[0]==nrx:return a.T
    raise ValueError('Unrecognized HDF5 pressure axes')

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--batch-size',type=int,default=128)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    with h5py.File(a.input) as h:
        sensors=np.asarray(h['sensor_mask_index'][...]).ravel();dt=float(h['dt'][...].ravel()[0])
    if sensors.size!=a.batch_size:raise ValueError('Need colocated full array input')
    env=dict(os.environ);env.pop('KWAVE_NATIVE_SOURCE_BATCH',None);env.setdefault('OMP_NUM_THREADS','4')
    # Explicit input order: caller can supply arbitrary physical TX ordering.
    source_env=os.environ.get('KWAVE_NATIVE_TX_INDICES_H5')
    sources=np.array([int(v) for v in source_env.split(',')]) if source_env else sensors
    times=[];serial=[]
    for j,source in enumerate(sources):
        ip=a.output_dir/f'serial_{j:03d}.h5';op=a.output_dir/f'serial_{j:03d}_out.h5'
        shutil.copyfile(a.input,ip)
        with h5py.File(ip,'r+') as h:h['p_source_index'][...]=source
        t=time.perf_counter();subprocess.run([str(a.binary.resolve()),'-i',str(ip),'-o',str(op),'-g',a.device,'--p_raw'],env=env,check=True)
        times.append(time.perf_counter()-t);serial.append(pressure(op,sensors.size))
    ip=a.output_dir/'native.h5';op=a.output_dir/'native_out.h5';expand_input(a.input,ip,a.batch_size)
    env['KWAVE_NATIVE_SOURCE_BATCH']='1';t=time.perf_counter()
    subprocess.run([str(a.binary.resolve()),'-i',str(ip),'-o',str(op),'-g',a.device,'--p_raw'],env=env,check=True)
    native_time=time.perf_counter()-t
    ref=np.stack(serial,axis=1);bat=pressure(op,sensors.size*a.batch_size).reshape(ref.shape)
    rf_error=float(np.linalg.norm(bat-ref)/np.linalg.norm(ref))
    f=np.arange(.3,1.0001,.025)*1e6
    if max(f)>=.5/dt:raise ValueError('Input cannot support the production 29 frequencies')
    kernel=np.exp(-2j*np.pi*f[:,None]*np.arange(ref.shape[0])[None,:]*dt)*dt
    dr=kernel@ref.reshape(ref.shape[0],-1);db=kernel@bat.reshape(ref.shape[0],-1)
    frequency_error=float(np.linalg.norm(db-dr)/np.linalg.norm(dr))
    passed=rf_error<1e-4 and frequency_error<1e-4
    result={'schema':'wfi.native_validation.v1','rf_relative_l2':rf_error,'dtft_relative_l2':frequency_error,
        'serial_wall_seconds':sum(times),'native_wall_seconds':native_time,'passed':passed,
        'speedup_including_process_io':sum(times)/native_time,'batch_size':a.batch_size,
        'note':'same modified binary, native disabled for serial; not a peak-memory measurement'}
    (a.output_dir/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
