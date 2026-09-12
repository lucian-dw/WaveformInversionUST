import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
native=load('native',ROOT/'native/run_native_source_batch.py')
runtime=load('runtime',ROOT/'python/wust_runtime.py')

class Contracts(unittest.TestCase):
    def make_input(self,path):
        with h5py.File(path,'w') as h:
            for key,val in {'Nx':8,'Ny':6,'Nz':1,'p_source_many':0,'absorbing_flag':0,'rho0':1000}.items():
                h[key]=np.array([[[val]]],dtype=np.uint64)
            h['sensor_mask_index']=np.array([[[3,7,11]]],dtype=np.uint64)
            h['p_source_index']=np.array([[[3]]],dtype=np.uint64)
            h['p_source_input']=np.arange(5,dtype=np.float32).reshape(5,1,1)
    def test_independent_sources_and_order(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'KWAVE_NATIVE_TX_INDICES_H5':'11,3,7'}):
            src=Path(d)/'a.h5';dst=Path(d)/'b.h5';self.make_input(src);before=src.read_bytes()
            native.expand_input(src,dst,3)
            self.assertEqual(before,src.read_bytes())
            with h5py.File(dst) as h:
                np.testing.assert_array_equal(h['p_source_index'][...].ravel(),[11,51,103])
                np.testing.assert_array_equal(h['sensor_mask_index'][...].ravel(),[3,7,11,51,55,59,99,103,107])
                self.assertEqual(h['p_source_input'].shape,(5,3,1))
    def test_reject_3d_absorption_and_duplicates(self):
        for key,value in [('Nz',2),('absorbing_flag',1)]:
            with tempfile.TemporaryDirectory() as d:
                src=Path(d)/'a.h5';self.make_input(src)
                with h5py.File(src,'r+') as h:h[key][...]=value
                with self.assertRaises(ValueError):native.expand_input(src,Path(d)/'b.h5',3)
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'KWAVE_NATIVE_TX_INDICES_H5':'3,3,7'}):
            src=Path(d)/'a.h5';self.make_input(src)
            with self.assertRaises(ValueError):native.expand_input(src,Path(d)/'b.h5',3)
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'a.h5';self.make_input(src)
            with self.assertRaises(ValueError):native.expand_input(src,src,3)

if __name__=='__main__':unittest.main()
