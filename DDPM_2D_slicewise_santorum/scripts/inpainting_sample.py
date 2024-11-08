import os
import torch as th
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import rotate

from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import create_model_and_diffusion


def get_model_and_diffusion(image_size):
    model_pt_path = "/scratch/santorum/checkpoints/replic_durrer_inpaint_slicewise_mni/savedmodel120000.pt"

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
        image_size=image_size,
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
        model_data = th.load(f, map_location="cpu")
    model.load_state_dict(model_data)
    return model, diffusion


def main(
    rank: int,
    use_gpu: bool,
    world_size: int,
    args: dict,
):
    if args.get("output_dir"):
        logger.configure(dir=args.get("output_dir"))
    else:
        logger.configure()

    dist_util.setup_dist(rank, world_size)

    # Set the device
    if use_gpu:
        device = th.device(f"cuda:{rank}")
        th.cuda.set_device(device)
    else:
        device = th.device(f"cpu:{rank}")
    
    logger.info(f"Device: {device}")

    actual_img_size = 224
    model_img_size = 256
    model, diffusion = get_model_and_diffusion(model_img_size)
    model.to(device)

    logger.info("Model and diffusion loaded")

    brats23_mni_dataset = BRATSDataset(
        "/scratch/santorum/bratsc2023-mni-dm-inpainting-preprocessed-3d/Training/",
        test_flag=True
    )

    logger.log(f"Loaded {len(brats23_mni_dataset)} samples from BRATS23-MNI dataset")

    sample_batch_2, sample_path_2, sample_slicedict_2 = brats23_mni_dataset[2]  # 140
    sample_batch_4, sample_path_4, sample_slicedict_4 = brats23_mni_dataset[4]  # 125
    sample_batch_5, sample_path_5, sample_slicedict_5 = brats23_mni_dataset[5]  # 110

    train_batch_to_repaint = th.stack(
        [
            sample_batch_2[..., 140],
            sample_batch_4[..., 125],
            sample_batch_5[..., 110],
        ],
        dim=0,
    )

    inpainted_train_batch, x_noisy_train_batch, original_train_batch = diffusion.p_sample_loop_known(
        model=model,
        shape=(train_batch_to_repaint.shape[0], 3, model_img_size, model_img_size),
        img=train_batch_to_repaint,
        clip_denoised=True,
        model_kwargs={},
        progress=True,
    )

    inpainted_train_batch = inpainted_train_batch.cpu()
    x_noisy_train_batch = x_noisy_train_batch.cpu()
    original_train_batch = original_train_batch.cpu()

    fig, ax = plt.subplots(
        nrows=len(inpainted_train_batch),
        ncols=5,
        figsize=(15, 3.3*len(inpainted_train_batch)),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.05]}
    )

    for i in range(len(inpainted_train_batch)):
        _voided_t1_img = rotate(original_train_batch[i, 0, ...].view(actual_img_size, actual_img_size), angle=-90)
        _ground_truth_img = rotate(original_train_batch[i, 2, ...].view(actual_img_size, actual_img_size), angle=-90)
        _inpainted_img = rotate(inpainted_train_batch[i].view(actual_img_size, actual_img_size), angle=-90)
        _diff_map_img = _inpainted_img - _ground_truth_img

        ax[i][0].imshow(_voided_t1_img, cmap="gray")
        ax[i][1].imshow(_ground_truth_img, cmap="gray")
        ax[i][2].imshow(_inpainted_img, cmap="gray")
        ax3_i = ax[i][3].imshow(_diff_map_img, norm=mpl.colors.CenteredNorm(), cmap="seismic")

        fig.colorbar(ax3_i, ax=ax[i][3], cax=ax[i][4])

        if i == 0:
            ax[i][0].set_title("MNI Voided GT")
            ax[i][1].set_title("MNI Ground-truth")
            ax[i][2].set_title("Inpainted")
            ax[i][3].set_title("Difference map")

    plt.tight_layout()
    os.makedirs("/scratch/santorum/inference/replic_durrer_inpaint_slicewise_mni/gpu", exist_ok=True)
    plt.savefig(
        "/scratch/santorum/inference/replic_durrer_inpaint_slicewise_mni/gpu/inpaint_on_training_samples.png",
    )


if __name__ == "__main__":
    main(rank=0, use_gpu=True, world_size=1, args={})
