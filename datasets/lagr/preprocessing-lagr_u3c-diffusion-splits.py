import h5py
import numpy as np

with h5py.File('Lagr_u3c_diffusion.h5', 'r') as h5f:
    rx0 = np.array(h5f.get('min'))
    rx1 = np.array(h5f.get('max'))
    x_train = (np.array(h5f.get('train'))+1)*(rx1-rx0)/2 + rx0

x_test  = x_train[-32768:].copy()
x_train = x_train[:-32768].copy()

rx0 = np.amin(x_train, axis=(0, 1))
rx1 = np.amax(x_train, axis=(0, 1))

x_train = 2*(x_train-rx0)/(rx1-rx0) - 1
x_test  = 2*(x_test-rx0) /(rx1-rx0) - 1

filename_out = 'Lagr_u3c_diffusion_splits.h5'
with h5py.File(filename_out, 'w') as hf:
    hf.create_dataset('min', data=rx0)
    hf.create_dataset('max', data=rx1)
    hf.create_dataset('train', data=x_train)
    hf.create_dataset('test',  data=x_test)
