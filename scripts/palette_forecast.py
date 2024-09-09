"""
Forecast a large batch of Lagrangian trajectories over time using 
a model and save them as a large numpy array.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.turb_datasets import load_data
from palette_diffusion.script_util import (
    palette_model_and_diffusion_defaults,
    palette_create_model_and_diffusion,
)
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = palette_create_model_and_diffusion(
        **args_to_dict(args, palette_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        deterministic=True,
    )

    logger.log("sampling...")
    all_images = []

    # Set global random seeds for all GPUs.
    seed = args.seed*4 + int(os.environ["CUDA_VISIBLE_DEVICES"])
    th.manual_seed(seed)

    # Set the forecasting window.
    zeros = th.zeros(
        (args.batch_size, args.in_channels, args.mask_size),
        dtype=th.float32,
        device=dist_util.dev(),
    )
    mask = th.zeros(
        (args.batch_size, args.in_channels, args.image_size),
        dtype=th.float32,
        device=dist_util.dev(),
    )
    mask[..., -args.mask_size:] = 1.0
    model_kwargs = {"mask": mask}

    while len(all_images) * args.batch_size < args.num_samples:
        batch, _ = next(data)
        batch = batch.to(dist_util.dev())
        preds = []
        for _ in range(args.num_predsteps):
            batch = th.cat([batch[..., args.mask_size:], zeros], dim=2)
            noise = th.randn_like(batch)
            model_kwargs["cond"] = (1.0 - mask) * batch + mask * noise
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args.batch_size, args.in_channels, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample = sample.clamp(-1, 1)
            preds.append(sample[..., -args.mask_size:])
            batch = (1.0 - mask) * batch + mask * sample
        sample = th.cat(preds, dim=2)
        sample = sample.permute(0, 2, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(
            logger.get_dir(), f"samples_{shape_str}-seed{args.seed:03d}.npz"
        )
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        dataset_path="",
        dataset_name="",
        mask_size=200,
        num_predsteps=10,
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        seed=0,
    )
    defaults.update(palette_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
