from guided_diffusion.script_util import create_model
from guided_diffusion import gaussian_diffusion as gd
from .tfg_diffusion import (
    BaseGuidanceDiffusion,
    DPSGuidanceDiffusion,
    TFGGuidanceDiffusion,
)


def tfg_diffusion_defaults():
    return dict(
        diffusion_steps=1000,
        sigma_small=False,
        noise_schedule="linear",
        predict_xstart=False,
        rescale_timesteps=False,
    )


def tfg_model_and_diffusion_defaults():
    res = dict(
        dims=2,
        image_size=64,
        in_channels=3,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(tfg_diffusion_defaults())
    return res


def guidance_defaults():
    return dict(
        guidance_name="",
        recur_steps=1,
        iter_steps=1,
        guidance_strength=1.0,
        rho=1.0,
        mu=1.0,
        sigma=0.01,
        eps_bsz=4,
        rho_schedule="increase",
        mu_schedule="increase",
        sigma_schedule="decrease",
    )


def tfg_create_model_and_diffusion(
    dims,
    image_size,
    in_channels,
    class_cond,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    sigma_small,
    noise_schedule,
    predict_xstart,
    rescale_timesteps,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    task,
    guidance_kwargs,
):
    model = create_model(
        dims,
        image_size,
        in_channels,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=False,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_tfg_diffusion(
        task=task,
        guidance_kwargs=guidance_kwargs,
        steps=diffusion_steps,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
    )
    return model, diffusion


def create_tfg_diffusion(
    *,
    task="",
    guidance_kwargs=None,
    steps=1000,
    sigma_small=False,
    noise_schedule="linear",
    predict_xstart=False,
    rescale_timesteps=False,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    model_mean_type = (
        gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
    )
    model_var_type = (
        gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL
    )
    loss_type = gd.LossType.MSE
    guidance_name = guidance_kwargs["guidance_name"]
    if guidance_name == "no":
        GuidanceDiffusion = BaseGuidanceDiffusion
    elif guidance_name == "dps":
        GuidanceDiffusion = DPSGuidanceDiffusion
    elif guidance_name == "tfg":
        GuidanceDiffusion = TFGGuidanceDiffusion
    else:
        raise NotImplementedError
    return GuidanceDiffusion(
        task=task,
        guidance_kwargs=guidance_kwargs,
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
