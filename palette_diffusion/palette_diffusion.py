import numpy as np
import torch as th

from continuous_diffusion.continuous_diffusion import GaussianDiffusionNoiseLevel


class PaletteDiffusion(GaussianDiffusionNoiseLevel):
    """
    This class is an adaptation of the GaussianDiffusionNoiseLevel class, tailored for 
    conditional diffusion models in image-to-image translation tasks.

    Reference: Saharia, C. et al. (2021). Palette: Image-to-Image Diffusion Models. 
    arXiv preprint arXiv:2111.05826.
    """

    def training_losses(self, model, x_start, t, model_kwargs, noise=None):
        """
        Compute training losses within the mask region.
        """
        mask = model_kwargs["mask"]
        if noise is None:
            noise = th.randn_like(x_start)

        sqrt_alpha_bar_sample, sqrt_one_minus_alpha_bar_sample = \
        self.sample_noise_level(t, x_start.shape)
        x_t = sqrt_alpha_bar_sample * x_start + sqrt_one_minus_alpha_bar_sample * noise

        terms = {}
        index = (slice(None),) + (0,) * (len(x_start.shape) - 1)
        model_output = model(x_t, sqrt_alpha_bar_sample[index], **model_kwargs)

        target = noise
        assert model_output.shape == target.shape == x_start.shape
        terms["mse"] = sum_flat(mask * (target - model_output) ** 2) / sum_flat(mask)
        terms["loss"] = terms["mse"]

        return terms


def sum_flat(tensor):
    """
    Sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, tensor.dim())))
