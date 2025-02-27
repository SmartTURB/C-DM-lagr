from palette_diffusion.palette_diffusion import sum_flat


def load_guider(task):
    """
    Returns log p_0(c|x_0) based on the specified task.

    For inpainting, p_0(c|x_0) = exp(-||cond - mask * x_0||^2), where cond is 
    the measurement, and mask = 1 in the measurement region and 0 elsewhere.
    """
    if task == "inpainting":
        def get_guidance(x0, guider_kwargs):
            cond = guider_kwargs["cond"]  # [B, C, H, ...]
            mask = guider_kwargs["mask"]  # [B, C, H, ...]
            return -sum_flat(mask * (cond - x0) ** 2)  / sum_flat(mask)  # [B]
    else:
        raise NotImplementedError
    return get_guidance
