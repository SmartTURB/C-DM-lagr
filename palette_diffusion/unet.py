import torch as th
from guided_diffusion.unet import UNetModel


class PaletteModel(UNetModel):
    """
    A UNetModel that performs image-to-image translation tasks.
    Expects additional kwargs `mask` and `cond`.

    :param mask: an [N x C x ...] Tensor specifying the target region.
    :param cond: an [N x C x ...] Tensor of the conditional input.
    """

    def __init__(self, image_size, in_channels, cond_channels, *args, **kwargs):
        super().__init__(image_size, in_channels + cond_channels, *args, **kwargs)

    def forward(self, x, timesteps, mask, cond, **kwargs):
        x = mask * x + (1.0 - mask) * cond
        x = th.cat([x, cond], dim=1)
        return super().forward(x, timesteps, **kwargs)
