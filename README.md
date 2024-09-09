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

### Dataset: 2D ocean drifters
