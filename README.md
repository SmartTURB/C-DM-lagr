# C-DM-lagr

This is the codebase for [Stochastic Reconstruction of Gappy Lagrangian Turbulent Signals by Conditional Generative Diffusion Models](https://arxiv.org/abs/2410.23971).

This repository is based on [SmartTURB/diffusion-lagr](https://github.com/SmartTURB/diffusion-lagr), with added functionality to perform **gappy Lagrangian turbulent signals reconstruction** conditioned on the measurements outside the gap. Specifically, two additional modules have been implemented:

- **[continuous_diffusion](./continuous_diffusion)**: Enables diffusion models to condition on a continuous noise level rather than discrete timesteps. See [WaveGrad](https://arxiv.org/abs/2009.00713) for details.
  
- **[palette_diffusion](./palette_diffusion)**: Enables conditional diffusion models for image-to-image translation tasks. See [Palette](https://arxiv.org/abs/2111.05826) for details.

## Installation

This codebase runs in a similar environment as [Development Environment](https://github.com/SmartTURB/diffusion-lagr#development-environment). Check [`env_setup.txt`](./env_setup.txt) for installation details with required packages and dependencies.

## Data Preparation

### Dataset: 3D HIT tracers

Please refer to [Preparing Data](https://github.com/SmartTURB/diffusion-lagr#preparing-data) for download and usage details of the file `Lagr_u3c_diffusion.h5`. Use the two scripts in [`datasets/lagr/`](./datasets/lagr/) to split the original dataset into 90% for training and 10% for testing for both the 1c and 3c cases.

### Dataset: 2D Ocean Drifters

One can access the hourly drifter data from the NOAA Global Drifter Program [here](https://www.aoml.noaa.gov/phod/gdp/hourly_data.php). We used version 2.01 and selected the file `gdp-v2.01.nc`.

To preprocess the data, including (1) dividing trajectories into non-overlapping 60-day segments and (2) removing segments with spurious points of high velocity or acceleration, run the script [`datasets/gdp1h/create-gdp1h_60d-datasets/create-gdp1h_60d-datasets.sh`](./datasets/gdp1h/create-gdp1h_60d-datasets/create-gdp1h_60d-datasets.sh), which requires the `clouddrift` package ([clouddrift.org](https://clouddrift.org/)). This will output two files: `gdp1h_60d-diffusion.h5`, containing the processed velocity segments, and `gdp1h_60d-pos0.h5`, containing the initial positions of these segments. Both files are available on the INFN Open Access Repository at [this link](https://doi.org/10.15161/oar.it/211740), and can be loaded as follows:

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

The `v2c` variable has a shape of `(115450, 1440, 2)`, representing 115,450 segments, each with 1,440 time instants (hours) and 2 velocity components. These velocities are min-max normalized with `rx0=-3` and `rx1=3` in the h5 file as the dataset `train`. `lon0` and `lat0` represent the initial longitude and latitude of the segments, each with a shape of `(115450)`.

To obtain the positions for a specific segment (e.g., `idx`), one can use the [`position_from_velocity`](https://clouddrift.org/_autosummary/clouddrift.kinematics.position_from_velocity.html#) function from the `clouddrift` package as follows:

```python
from clouddrift.kinematics import position_from_velocity

ve_idx, vn_idx = v2c[idx, :, 0], v2c[idx, :, 1]
time = np.arange(1440) * 3600  # units: seconds
lon_idx, lat_idx = position_from_velocity(ve_idx, vn_idx, time, lon0[idx], lat0[idx])
```

## Training

Please refer to the parent repository’s [Training section](https://github.com/SmartTURB/diffusion-lagr#training) for detailed information, including hyperparameter configuration. The only additional flag in this case is `--mask_mode`, which has the following options:

- `center1d<lg>`: Specifies a central gap of size `<lg>`.
- `right1d<lg>`: Specifies a right-end gap of size `<lg>`.
- `interp1d<scale_factor>`: Specifies a sample point every `<scale_factor>` points for interpolation cases.

See the function `get_mask` in [`palette_diffusion/palette_datasets.py`](./palette_diffusion/palette_datasets.py#L87) for customizing the reconstruction scenario.

For Lagrangian turbulence reconstruction with a central gap of size $50\tau_\eta$, use the following flags:

```bash
DATA_FLAGS="--mask_mode center1d500 --dataset_path datasets/lagr/Lagr_u3c_diffusion_splits.h5 --dataset_name train"
MODEL_FLAGS="--dims 1 --image_size 2000 --in_channels 3 --num_channels 128 --num_res_blocks 3 --attention_resolutions 250,125 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule tanh6,1"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64"
```

For ocean drifter observation reconstruction with a central gap of 360 hours, use the following flags:

```bash
DATA_FLAGS="--mask_mode center1d360 --dataset_path datasets/gdp1h/gdp1h_v2c_diffusion_splits.h5 --dataset_name train"
MODEL_FLAGS="--dims 1 --image_size 1440 --in_channels 2 --num_channels 128 --num_res_blocks 3 --attention_resolutions 180,90 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule tanh6,1"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64"
```

Use [`scripts/palette_train.py`](./scripts/palette_train.py) to train the conditional diffusion model:

```bash
mpiexec -n $NUM_GPUS python scripts/palette_train.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

## Reconstructing

Please refer to the parent repository’s [Sampling section](https://github.com/SmartTURB/diffusion-lagr#sampling) for detailed information. The only additional option here is `--seed`, which sets the random seed for reconstruction.

For Lagrangian turbulence reconstruction with a central gap of size $50\tau_\eta$, use the following flags:

```bash
DATA_FLAGS="--mask_mode center1d500 --dataset_path datasets/lagr/Lagr_u3c_diffusion_splits.h5 --dataset_name test"
MODEL_FLAGS="--dims 1 --image_size 2000 --in_channels 3 --num_channels 128 --num_res_blocks 3 --attention_resolutions 250,125 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule tanh6,1"
SAMPLE_FLAGS="--num_samples 32768 --batch_size 64 --model_path /path/to/model.pt --seed 0"
```

For ocean drifter observation reconstruction with a central gap of 360 hours, use the following flags:

```bash
DATA_FLAGS="--mask_mode center1d360 --dataset_path datasets/gdp1h/gdp1h_v2c_diffusion_splits.h5 --dataset_name test"
MODEL_FLAGS="--dims 1 --image_size 1440 --in_channels 2 --num_channels 128 --num_res_blocks 3 --attention_resolutions 180,90 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule tanh6,1"
SAMPLE_FLAGS="--num_samples 11545 --batch_size 64 --model_path /path/to/model.pt --seed 0"
```

Use [`scripts/palette_sample.py`](./scripts/palette_sample.py) to reconstruct the test data:

```bash
python scripts/palette_sample.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
```
