# C-DM-lagr

This is the code base for [Stochastic Reconstruction of Gappy Lagrangian Turbulent Signals by Conditional Generative Diffusion Models](arxiv_link_placeholder).

This repository is based on [SmartTURB/diffusion-lagr](https://github.com/SmartTURB/diffusion-lagr), with added functionality to perform **Gappy Lagrangian Turbulent Signals reconstruction** conditioned on the measurements outside the gap. Specifically, two additional modules have been implemented:

- **continuous_diffusion**: Enables diffusion models to condition on a continuous noise level instead of discrete timesteps. The implementation follows the method described in:
  > Chen, N. et al. (2020). *WaveGrad: Estimating Gradients for Waveform Generation*. arXiv preprint arXiv:2009.00713.

- **palette_diffusion**: Enables conditional diffusion models for image-to-image translation tasks. Reference:
  > Saharia, C. et al. (2021). *Palette: Image-to-Image Diffusion Models*. arXiv preprint arXiv:2111.05826.

## Installation

This codebase runs in a similar environment as [Development Environment](https://github.com/SmartTURB/diffusion-lagr#development-environment). Check [`env_setup.txt`](./env_setup.txt) for installation details with required packages and dependencies.

## Data Preparation

### Dataset: 3D HIT tracers

Please refer to [Preparing Data](https://github.com/SmartTURB/diffusion-lagr#preparing-data) for download and usage details of the file `Lagr_u3c_diffusion.h5`. Use the two scripts in [`datasets/lagr/`](./datasets/lagr/) to split the original dataset into 90% for training and 10% for testing for both the 1c and 3c cases.

### Dataset: 2D Ocean Drifters

One can access the hourly drifter data at [this link](https://www.aoml.noaa.gov/phod/gdp/hourly_data.php). We used version 2.01 and selected the file `gdp-v2.01.nc`.

To preprocess the data, including (1) dividing trajectories into non-overlapping 60-day segments and (2) removing segments with spurious points of high velocity or acceleration, run the script [`datasets/gdp1h/create-gdp1h_60d-datasets/create-gdp1h_60d-datasets.sh`](./datasets/gdp1h/create-gdp1h_60d-datasets/create-gdp1h_60d-datasets.sh), which requires the `clouddrift` package ([clouddrift.org](https://clouddrift.org/)). This will output two files: `gdp1h_60d-diffusion.h5` and `gdp1h_60d-pos0.h5`, which can be loaded as follows:

```python
import h5py
import numpy as np

with h5py.File('gdp1h_60d-diffusion.h5', 'r') as h5f:
    rx0 = np.array(h5f.get('min'))
    rx1 = np.array(h5f.get('max'))
    v2c = (np.array(h5f.get('train'))+1)*(rx1-rx0)/2 + rx0

with h5py.File('gdp1h_60d-pos0.h5', 'r') as h5f:
    lon0 = np.array(h5f.get('lon0'))
    lat0 = np.array(h5f.get('lat0'))
```

The `v2c` variable has a shape of `(115450, 1440, 2)`, representing 115,450 segments, each with 1,440 time instants (hours) and 2 velocity components. These velocities are min-max normalized with `rx0=-3` and `rx1=3` in the h5 file as the dataset `train`. `lon0` and `lat0` are two arrays, each of shape `(115450)`, corresponding to the initial longitude and latitude of each segment.

To compute the positions for a specific segment (e.g., `idx`), one can use the [`clouddrift`](https://clouddrift.org/) API as follows:

```python
from clouddrift.kinematics import position_from_velocity

ve_idx, vn_idx = v2c[idx, :, 0], v2c[idx, :, 1]
time = np.arange(1440) * 3600  # units: seconds
lon_idx, lat_idx = position_from_velocity(ve_idx, vn_idx, time, lon0[idx], lat0[idx])
```
