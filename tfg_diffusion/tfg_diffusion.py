import numpy as np
import torch as th

from guided_diffusion.gaussian_diffusion import ModelVarType, _extract_into_tensor
from continuous_diffusion.continuous_diffusion import GaussianDiffusionNoiseLevel
from .tasks import load_guider


class BaseGuidanceDiffusion(GaussianDiffusionNoiseLevel):
    """
    Base class for a series of training-free guidance methods designed to generate 
    samples with desirable target properties without requiring additional training.

    This class provides a foundation for implementing various guidance strategies 
    applicable to diffusion models.

    :param task: str, the name of the task used to load the `get_guidance` function.
    :param guidance_kwargs: dict, hyperparameters for guidance strategies.
    :param kwargs: Additional keyword arguments to create the base diffusion process.
    """

    def __init__(self, task, guidance_kwargs, **kwargs):
        self.get_guidance = load_guider(task)
        self.guidance_kwargs = guidance_kwargs
        super().__init__(**kwargs)

    def q_sample_prev(self, x_prev, t, noise=None):
        """
        Diffuse the data for one diffusion step.

        In other words, sample from q(x_t | x_{t-1}).

        :param x_prev: the data batch at t-1.
        :param t: the indices of diffusion steps (minus 1). Here, 0 means the first step.
        :param noise: if specified, the split-out normal noise.
        :return: the noisy data batch at t.
        """
        if noise is None:
            noise = th.randn_like(x_prev)
        assert noise.shape == x_prev.shape
        beta = _extract_into_tensor(self.betas, t, x_prev.shape)
        return th.sqrt(1 - beta) * x_prev + th.sqrt(beta) * noise

    def _predict_xstart(self, x_t, t, eps, clip_denoised=True, denoised_fn=None):
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        return process_xstart(self._predict_xstart_from_eps(x_t, t, eps))

    def _predict_x_prev_from_eps(self, x, t, eps, clip_denoised=True, denoised_fn=None):
        """
        Sample x_{t-1} from the current x_t (x) and the predicted noise (eps).
        """
        model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE:
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ModelVarType.FIXED_SMALL:
                self.posterior_log_variance_clipped,
        }[self.model_var_type]
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        pred_xstart = self._predict_xstart(x, t, eps, clip_denoised, denoised_fn)
        model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)

        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        return model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise

    def _wrapped_model(self, model, x, t, **kwargs):
        sqrt_alpha_bar = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape)
        index = (slice(None),) + (0,) * (len(x.shape) - 1)
        return model(x, sqrt_alpha_bar[index], **kwargs)

    def guide_step(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, 
        guider_kwargs=None
    ):
        """
        Sample x_{t-1} from the model with guidance at the given timestep, 
        ensuring the predictions are towards the desired target properties.

        :param guider_kwargs: a dictionary of additional keyword arguments 
            to pass to the `self.get_guidance` function.
        """
        for recur_step in range(self.guidance_kwargs["recur_steps"]):

            # predict epsilon with the model
            eps = self._wrapped_model(model, x, t, **model_kwargs)

            # predict x_0 using x_t and epsilon
            x0 = self._predict_xstart(x, t, eps, clip_denoised, denoised_fn)

            # sample x_{t-1} using x_t and epsilon
            x_prev = self._predict_x_prev_from_eps(x, t, eps, clip_denoised, denoised_fn)

            # resample x_t from q(x_t | x_{t-1})
            x = self.q_sample_prev(x_prev, t)

        return x_prev

    def sample(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        guider_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the unconditional model with desired target properties 
        using training-free guidance.

        :param guider_kwargs: A dictionary of additional keyword arguments to pass to 
            the `self.guide_step` function.
        Other arguments are the same as in `guided_diffusion.GaussianDiffusion.p_sample_loop()`.
        :return: A non-differentiable batch of samples.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                img = self.guide_step(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    guider_kwargs=guider_kwargs,
                )
        return img


class DPSGuidanceDiffusion(BaseGuidanceDiffusion):
    """
    Reference: Chung, H. et al. (2022). Diffusion posterior sampling for general 
    noisy inverse problems. arXiv preprint arXiv:2209.14687.
    """

    def guide_step(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, 
        guider_kwargs=None
    ):
        with th.enable_grad():
            x_g = x.clone().detach().requires_grad_()

            # predict epsilon with the model
            eps = self._wrapped_model(model, x_g, t, **model_kwargs)

            # predict x_0 using x_t and epsilon
            x0 = self._predict_xstart(x_g, t, eps, clip_denoised, denoised_fn)

            # compute DPS guidance on x_t
            log_probs = self.get_guidance(x0, guider_kwargs)
            guidance = th.autograd.grad(log_probs.sum(), x_g)[0]

        # follow the schedule of DPS paper
        guidance /= 2 * th.sqrt(-log_probs).view(-1, *([1] * (len(x_g.shape) - 1)))
        guidance *= self.guidance_kwargs["guidance_strength"]

        # sample x_{t-1} using x_t and epsilon
        x_prev = self._predict_x_prev_from_eps(x, t, eps, clip_denoised, denoised_fn)
        return x_prev + guidance


class TFGGuidanceDiffusion(BaseGuidanceDiffusion):
    """
    Reference: Ye, H. et al. (2024). TFG: Unified Training-Free Guidance for 
    Diffusion Models. arXiv preprint arXiv:2409.15761.
    """

    def guide_step(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, 
        guider_kwargs=None
    ):
        pass
