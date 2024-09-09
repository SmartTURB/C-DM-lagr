import h5py
import numpy as np
from clouddrift.kinematics import velocity_from_position

vit  = 3  # velocity threshold, units: m/s
dvit = 4  # velocity increment threshold

filename_in = 'dataset-60days.h5'
with h5py.File(filename_in, 'r') as hf:
    lon = np.array(hf['lon'])
    lat = np.array(hf['lat'])
    ve = np.array(hf['ve'])
    vn = np.array(hf['vn'])

ol_vi_idxs = np.where((np.abs(ve) > vit) | (np.abs(vn) > vit))
ol_vi_idxs = np.unique(ol_vi_idxs[0])

time = np.arange(lon.shape[-1])*3600
u, v = velocity_from_position(lon, lat, time)

ol_ui_idxs = np.where((np.abs(u) > vit) | (np.abs(v) > vit))
ol_ui_idxs = np.unique(ol_ui_idxs[0])

dve = ve[:, 1:] - ve[:, :-1]
dvn = vn[:, 1:] - vn[:, :-1]

ol_dvi_idxs = np.where((np.abs(dve) > dvit) | (np.abs(dvn) > dvit))
ol_dvi_idxs = np.unique(ol_dvi_idxs[0])

ol_mask = np.zeros(len(lon), dtype=bool)
ol_mask[ol_vi_idxs] = True
ol_mask[ol_ui_idxs] = True
ol_mask[ol_dvi_idxs] = True

filename_out = f'{filename_in[:-3]}-vit3-dvit4.h5'
with h5py.File(filename_out, 'w') as hf:
    hf.create_dataset('ol_vi_idxs', data=ol_vi_idxs)
    hf.create_dataset('ol_ui_idxs', data=ol_ui_idxs)
    hf.create_dataset('ol_dvi_idxs', data=ol_dvi_idxs)
    hf.create_dataset('lon', data=lon[~ol_mask])
    hf.create_dataset('lat', data=lat[~ol_mask])
    hf.create_dataset('ve', data=ve[~ol_mask])
    hf.create_dataset('vn', data=vn[~ol_mask])
