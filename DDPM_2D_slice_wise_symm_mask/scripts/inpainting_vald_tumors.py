import os
import math
import numpy as np
import torch
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import create_model_and_diffusion


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


def plot_save_slices(
    original_batch: torch.Tensor,
    inpainted_batch: torch.Tensor,
    actual_img_size: int,
    slice_indices: list,
    subject_name: str,
    output_dir: str,
):
    # Plot the original and inpainted slices
    for i in range(inpainted_batch.shape[0]):
        n_figs = original_batch.shape[1] + 2 # channels, inpainted, diff map
        width_ratios = ([1] * n_figs) + [0.05]
        fig, axs = plt.subplots(1, n_figs+1, figsize=(3*n_figs, 3.5), gridspec_kw={"width_ratios": width_ratios})
        for k in range(original_batch.shape[1]):
            _img_show = original_batch[i,k,...].view(actual_img_size, actual_img_size).numpy()
            axs[k].imshow(_img_show, cmap="gray")
            axs[k].set_title(f"Channel {k}")
        
        inpainted_img = inpainted_batch[i].view(actual_img_size, actual_img_size).numpy()
        axs[-3].imshow(inpainted_img, cmap="gray")
        axs[-3].set_title("Inpainted")

        groundtruth_img = original_batch[i,-1,...].view(actual_img_size, actual_img_size).numpy()
        diff_map = inpainted_img - groundtruth_img
        ax_cb = axs[-2].imshow(diff_map, norm=mpl.colors.CenteredNorm(), cmap="seismic")
        axs[-2].set_title("Diff Map")

        fig.colorbar(ax_cb, ax=axs[-2], cax=axs[-1])

        plt.savefig(os.path.join(output_dir, f"{subject_name}_slice_{slice_indices[i]}.png"))


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

    set_seed(0)

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

    # Load the dataset
    brats_dataset = BRATSDataset(
        args.data_dir,
        test_flag=True,
        override_seqtypes=["unhealthy-voided", "symm-unhealthy-mask", "t1n"],
        ref_mask="mask-unhealthy",
    )

    if len(brats_dataset) == 0:
        raise ValueError(f"No samples found in the dataset in {args.data_dir}")

    logger.info(f"Loaded {len(brats_dataset)} samples from BRATS dataset")

    for i in range(len(brats_dataset)):
        filedict_i = brats_dataset.database[i]
        if "BraTS-GLI-0166" not in filedict_i["t1n"]:
            continue

        logger.info(f"Inpainting slices of {os.path.basename(filedict_i['t1n']).replace('.nii.gz', '')}")
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

            # Save the inpainted and original slices
            if args.png_output_dir:
                os.makedirs(args.png_output_dir, exist_ok=True)
                plot_save_slices(
                    original_batch=original_batch_i_j,
                    inpainted_batch=inpainted_batch_i_j,
                    actual_img_size=args.actual_image_size,
                    slice_indices=slicedict_i_j,
                    subject_name=os.path.basename(path_i).replace(".nii.gz", ""),
                    output_dir=args.png_output_dir,
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_pt_path", type=str)
    parser.add_argument("--model_image_size", type=int)
    parser.add_argument("--actual_image_size", type=int)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--png_output_dir", type=str, default=None)

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