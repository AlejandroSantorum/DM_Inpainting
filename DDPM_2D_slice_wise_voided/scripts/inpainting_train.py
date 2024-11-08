"""
Train a diffusion model on images.
"""
import os
import sys
import argparse
import random

sys.path.append("..")
sys.path.append(".")
from datetime import datetime
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset
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

    if args.seqtypes is not None:
        # seqtypes example: "voided,mask,t1n"
        override_seqtypes = args.seqtypes.split(",")
        logger.log("Overriding seqtypes to: " + str(override_seqtypes))
    else:
        override_seqtypes = None

    logger.log(f"Creating data loader with data_dir '{args.data_dir}'")
    ds = BRATSDataset(
        args.data_dir,
        test_flag=False,
        override_seqtypes=override_seqtypes,
        ref_mask=args.ref_mask,
        max_samples=args.max_samples,
        seed=args.bratsloader_seed,
    )

    # Create a distributed sampler
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank)

    datal = th.utils.data.DataLoader(ds, batch_size=args.batch_size, sampler=sampler)
    # data = iter(datal)

    if args.training_seed is not None:
        set_seed(int(args.training_seed))

    logger.log("Initiating training ...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=datal,  # not used
        dataloader=datal,
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
        seqtypes=None,  # NOTE: Added by Santorum
        ref_mask="mask",  # NOTE: Added by Santorum
        max_samples=None,  # NOTE: Added by Santorum
        bratsloader_seed=None,  # NOTE: Added by Santorum
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
