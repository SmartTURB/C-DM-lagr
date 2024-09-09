import xarray as xr
import numpy as np
import h5py
from clouddrift.ragged import segment, apply_ragged, chunk

filename = 'gdp-v2.01.nc'
ds = xr.open_dataset(filename, engine='netcdf4', decode_times=True)

# define a new rowsize variable for all contiguous hourly segments
dt = np.timedelta64(3600, 's')
rowsize = segment(ds.time, dt, rowsize=ds.rowsize)

# shuffle
rng = np.random.default_rng(seed=0)
idx = None

sixty_days = 60*24
dnames = ['lon', 'lat', 've', 'vn']

with h5py.File('dataset-60days.h5', 'w') as hf:
    for dname in dnames:
        data = apply_ragged(chunk, ds[dname], rowsize, \
            length=sixty_days, overlap=0, align='middle')
        if idx is None:
            idx = rng.permutation(data.shape[0])
        hf.create_dataset(dname, data=data[idx])
