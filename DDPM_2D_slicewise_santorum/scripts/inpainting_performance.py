import os
import math
import random
import torch

from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import create_model_and_diffusion

import numpy as np
from utils.metrics import mse_2d, snr_2d, psnr_2d
from skimage.metrics import structural_similarity


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_model_and_diffusion(model_image_size: int, model_pt_path: str):
    model, diffusion = create_model_and_diffusion(
        # diffusion defaults
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        # model defaults
        image_size=model_image_size,
        num_channels=128,  # why 128 channels?
        num_res_blocks=2,
        num_heads=1,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    with open(model_pt_path, "rb") as f:
        model_data = torch.load(f, map_location="cpu")
    model.load_state_dict(model_data)
    return model, diffusion


def main(
    rank: int,
    use_gpu: bool,
    world_size: int,
    args: dict,
):
    # Configure the logger
    if args.output_dir:
        logger.configure(dir=args.output_dir)
    else:
        logger.configure()
    
    set_seed(0)
    
    # Check if the required arguments are present
    if "data_dir" not in args:
        raise ValueError("data_dir not found in args")
    if "model_image_size" not in args:
        raise ValueError("model_image_size not found in args")
    if "actual_image_size" not in args:
        raise ValueError("actual_image_size not found in args")
    if "model_pt_path" not in args:
        raise ValueError("model_pt_path not found in args")
    if "sample_batch_size" not in args:
        raise ValueError("batch_size_vol not found in args")

    logger.info(f"Input Arguments: {args}")

    # Set the distributed configuration
    dist_util.setup_dist(rank, world_size)

    # Set up the device (GPU or CPU)
    if use_gpu:
        logger.info(f"Using GPU {rank}")
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        logger.info(f"Using CPU {rank}")
        device = torch.device(f"cpu:{rank}")
    
    logger.info(f"Device: {device}")

    # Load the model and diffusion
    model, diffusion = get_model_and_diffusion(
        model_image_size=args.model_image_size, model_pt_path=args.model_pt_path
    )
    model.to(device)
    logger.info("Model and diffusion loaded")

    if args.override_seqtypes is not None:
        # seqtypes example: "voided,mask,t1n"
        override_seqtypes = args.override_seqtypes.split(",")
        logger.log("Overriding seqtypes to: " + str(override_seqtypes))
    else:
        override_seqtypes = None

    # Load the dataset
    brats_dataset = BRATSDataset(
        args.data_dir,
        test_flag=True,
        override_seqtypes=override_seqtypes,
        ref_mask=args.ref_mask,
        max_samples=args.max_samples,
        seed=args.bratsloader_seed,
    )

    if len(brats_dataset) == 0:
        raise ValueError(f"No samples found in the dataset in {args.data_dir}")

    logger.info(f"Loaded {len(brats_dataset)} samples from BRATS dataset")

    # Lists to store the performance metrics
    mse_list, snr_list, psnr_list, ssim_list = [], [], [], []

    for i in range(len(brats_dataset)):
        filedict_i = brats_dataset.database[i]
        # if "BraTS-GLI-0166" not in filedict_i["t1n"]:
        #     continue

        logger.info(f"Inpainting slices of image no. {i + 1} ...")
        batch_i, path_i, slicedict_i = brats_dataset[i]

        num_p_sample_loop_iters = math.ceil(len(slicedict_i) / args.sample_batch_size)

        for j in range(num_p_sample_loop_iters):
            logger.info(f"\tProcessing batch no. {j + 1} of {num_p_sample_loop_iters} ...")

            # get the indices for the current batch
            start_idx = j * args.sample_batch_size
            end_idx = start_idx + args.sample_batch_size
            # get the slice indices to inpaint in the current batch
            slicedict_i_j = slicedict_i[start_idx:end_idx]
            # get the batch of images to inpaint based on the slice indices
            batch_i_j = batch_i[:,:,:,slicedict_i_j]
            # permute the dimensions to match the model's input shape (batch size, channels, height, width)
            batch_i_j = torch.permute(batch_i_j, (3, 0, 1, 2))

            # perform inpainting on the current batch of images using the DDPM model
            inpainted_batch_i_j, x_noisy_batch_i_j, original_batch_i_j = diffusion.p_sample_loop_known(
                model=model,
                shape=(batch_i_j.shape[0], batch_i_j.shape[1], args.model_image_size, args.model_image_size),
                img=batch_i_j,
                clip_denoised=True,
                model_kwargs={},
                progress=True,
            )

            inpainted_batch_i_j = inpainted_batch_i_j.cpu()
            x_noisy_batch_i_j = x_noisy_batch_i_j.cpu()
            original_batch_i_j = original_batch_i_j.cpu()

            # calculate the performance metrics for the inpainted images
            for k in range(inpainted_batch_i_j.shape[0]):
                inpainted_slice_k = inpainted_batch_i_j[k].view(args.actual_image_size, args.actual_image_size).numpy()
                groundtruth_slice_k = original_batch_i_j[k,-1,...].view(args.actual_image_size, args.actual_image_size).numpy()

                mse_k = mse_2d(test_img=inpainted_slice_k, ref_img=groundtruth_slice_k)
                snr_k = snr_2d(test_img=inpainted_slice_k, ref_img=groundtruth_slice_k)
                psnr_k = psnr_2d(test_img=inpainted_slice_k, ref_img=groundtruth_slice_k)
                ssim_k = structural_similarity(inpainted_slice_k, groundtruth_slice_k, data_range=1)

                mse_list.append(mse_k)
                snr_list.append(snr_k)
                psnr_list.append(psnr_k)
                ssim_list.append(ssim_k)

    # Calculate the performance metrics
    mse_list = np.array(mse_list)
    snr_list = np.array(snr_list)
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)

    logger.info(f"Dropping {np.sum(np.isnan(mse_list))} NaN values from MSE array")
    pr_mse_list = mse_list[~np.isnan(mse_list)]
    logger.info(f"Dropping {np.sum(np.isnan(snr_list))} NaN values from SNR array")
    pr_snr_list = snr_list[~np.isnan(snr_list)]
    logger.info(f"Dropping {np.sum(np.isnan(psnr_list))} NaN values from PSNR array")
    pr_psnr_list = psnr_list[~np.isnan(psnr_list)]
    logger.info(f"Dropping {np.sum(np.isnan(ssim_list))} NaN values from SSIM array")
    pr_ssim_list = ssim_list[~np.isnan(ssim_list)]

    logger.info("====================================")
    logger.info("Mean Performance Metrics:")
    logger.info(f"MSE: {np.mean(pr_mse_list)} ± {np.std(pr_mse_list)}")
    logger.info(f"SNR: {np.mean(pr_snr_list)} ± {np.std(pr_snr_list)}")
    logger.info(f"PSNR: {np.mean(pr_psnr_list)} ± {np.std(pr_psnr_list)}")
    logger.info(f"SSIM: {np.mean(pr_ssim_list)} ± {np.std(pr_ssim_list)}")
    logger.info("====================================")
    logger.info("Quantiles of Performance Metrics:")
    for quantile in [0.25, 0.5, 0.75]:
        logger.info(f"MSE {quantile}: {np.quantile(pr_mse_list, quantile)}")
        logger.info(f"SNR {quantile}: {np.quantile(pr_snr_list, quantile)}")
        logger.info(f"PSNR {quantile}: {np.quantile(pr_psnr_list, quantile)}")
        logger.info(f"SSIM {quantile}: {np.quantile(pr_ssim_list, quantile)}")
    logger.info("====================================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_pt_path", type=str)
    parser.add_argument("--model_image_size", type=int)
    parser.add_argument("--actual_image_size", type=int)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--override_seqtypes", type=str, default=None)
    parser.add_argument("--ref_mask", type=str, default="mask")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--bratsloader_seed", type=int, default=None)

    args = parser.parse_args()

    # number of GPUs available
    world_size = torch.cuda.device_count()
    print(f"Number of CUDA available devices (world size): {world_size}")
    print(f"IDs of CUDA available devices: {os.getenv('CUDA_VISIBLE_DEVICES')}")

    if world_size > 0:
        torch.multiprocessing.spawn(
            main,
            args=(True, world_size, args),
            nprocs=world_size,
            join=True,
        )
    else:
        main(rank=0, use_gpu=False, world_size=1, args=args)
