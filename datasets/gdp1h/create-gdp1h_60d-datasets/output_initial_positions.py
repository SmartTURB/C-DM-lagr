import h5py
import numpy as np

filename_in = 'dataset-60days-vit3-dvit4.h5'
with h5py.File(filename_in, 'r') as hf:
    lon0 = np.array(hf['lon'][:, 0])
    lat0 = np.array(hf['lat'][:, 0])

filename_out = f'gdp1h_60d-pos0.h5'
with h5py.File(filename_out, 'w') as hf:
    hf.create_dataset('lon0', data=lon0)
    hf.create_dataset('lat0', data=lat0)
