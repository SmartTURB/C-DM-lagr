import h5py
import numpy as np

with h5py.File('create-gdp1h_60d-datasets/gdp1h_60d-diffusion.h5', 'r') as h5f:
    rx0 = np.array(h5f.get('min'))
    rx1 = np.array(h5f.get('max'))
    x_train = np.array(h5f['train'])

x_test  = x_train[-11545:].copy()
x_train = x_train[:-11545].copy()

filename_out = 'gdp1h_v2c_diffusion_splits.h5'
with h5py.File(filename_out, 'w') as hf:
    hf.create_dataset('min', data=rx0)
    hf.create_dataset('max', data=rx1)
    hf.create_dataset('train', data=x_train)
    hf.create_dataset('test',  data=x_test)
