"""
Train a diffusion model on images.
"""
import os
import sys
import argparse
import random

sys.path.append(".")
sys.path.append("..")
from datetime import datetime
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.brain_dataset import BrainDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import numpy as np
import torch as th
from torch.utils.data.distributed import DistributedSampler
from guided_diffusion.train_util import TrainLoop


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)


def train_model(
    rank: int,
    use_gpu: bool,
    world_size: int,
    args: dict,
):
    if args.output_dir is not None:
        logger.configure(dir=args.output_dir)
    else:
        logger.configure()

    dist_util.setup_dist(rank, world_size)

    # Set the device
    if use_gpu:
        device = th.device(f"cuda:{rank}")
        th.cuda.set_device(device)
    else:
        device = th.device(f"cpu:{rank}")

    logger.log(f"creating model and diffusion using device {device} ...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, maxt=1000
    )

    if args.input_img_types is not None:
        input_img_types = args.input_img_types.split(",")
        logger.log("Using input image types: " + str(input_img_types))
    else:
        input_img_types = None
    
    if args.output_img_types is not None:
        output_img_types = args.output_img_types.split(",")
        logger.log("Using output image types: " + str(output_img_types))
    else:
        output_img_types = None

    logger.log(f"Creating brain dataset loading from '{args.data_dir}'")
    brain_dataset = BrainDataset(
        directory=args.data_dir,
        test_flag=False,  # training
        input_img_types=input_img_types,
        output_img_types=output_img_types,
        reference_img_type=args.reference_img_type,
        num_cutoff_samples=args.num_cutoff_samples,
        num_max_samples=args.num_max_samples,
        seed=args.dataset_seed,
    )

    # Create a distributed sampler
    sampler = DistributedSampler(brain_dataset, num_replicas=world_size, rank=rank)

    dataloader = th.utils.data.DataLoader(
        brain_dataset, batch_size=args.batch_size, sampler=sampler
    )

    if args.training_seed is not None:
        set_seed(int(args.training_seed))

    logger.log("Initiating training ...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=dataloader,  # not used
        dataloader=dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

    logger.log("Training finished.")


def main():
    input_args = create_argparser().parse_args()

    print("Training " + str(datetime.now()))
    print("Input args: " + str(input_args))
    # number of GPUs
    world_size = th.cuda.device_count()
    print(f"Number of CUDA available devices (world size): {world_size}")
    print(f"IDs of CUDA available devices: {os.getenv('CUDA_VISIBLE_DEVICES')}")

    if world_size > 0:
        th.multiprocessing.spawn(
            train_model,
            args=(True, world_size, input_args),
            nprocs=world_size,
            join=True,
        )
    else:
        train_model(rank=0, use_gpu=False, world_size=1, args=input_args)


def create_argparser():
    defaults = dict(
        data_dir="",
        output_dir="",  # NOTE: Added by Santorum
        input_img_types=None,  # NOTE: Added by Santorum
        output_img_types=None,  # NOTE: Added by Santorum
        reference_img_type="mask",  # NOTE: Added by Santorum
        num_cutoff_samples=None,  # NOTE: Added by Santorum
        num_max_samples=None,  # NOTE: Added by Santorum
        dataset_seed=None,  # NOTE: Added by Santorum
        training_seed=None,  # NOTE: Added by Santorum
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
