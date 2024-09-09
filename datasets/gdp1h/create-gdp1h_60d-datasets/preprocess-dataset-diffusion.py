import h5py
import numpy as np

filename_in = 'dataset-60days-vit3-dvit4.h5'
with h5py.File(filename_in, 'r') as hf:
    ve = np.array(hf['ve'])
    vn = np.array(hf['vn'])

rx0, rx1 = -3, 3
x_train = np.stack((ve, vn), axis=-1)
x_train = 2*(x_train-rx0)/(rx1-rx0) - 1

filename_out = f'{filename_in[:-3]}-diffusion.h5'
with h5py.File(filename_out, 'w') as hf:
    hf.create_dataset('min', data=rx0)
    hf.create_dataset('max', data=rx1)
    hf.create_dataset('train', data=x_train)
