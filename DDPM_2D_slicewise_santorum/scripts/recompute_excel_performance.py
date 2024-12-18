import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append("/home/santorum/phd")
from utils.metrics import mse_2d, snr_2d, psnr_2d, ssim_2d


INFERENCE_SLICES_BASE_PATH = "/scratch/santorum/inference"
STORE_EXCEL_BASE_PATH = "/home/santorum/phd/results/ixi_durrer_slicewise"


def recompute_metrics_for_checkpoint(model_name: str, checkpoint_name: str, checkpoint_path: str):
    inpainted_slices_path = os.path.join(checkpoint_path, "validation_slices", "inpainted")
    groundtruth_slices_path = os.path.join(checkpoint_path, "validation_slices", "groundtruth")
    masks_slices_path = os.path.join(checkpoint_path, "validation_slices", "ref_mask")

    subject_names = []
    slice_indices = []
    mse_list = []
    snr_list = []
    psnr_list = []
    ssim_list = []

    for slice_name in tqdm(os.listdir(inpainted_slices_path), desc="Recomputing metrics", file=sys.stdout):
        inpainted_slice = np.load(os.path.join(inpainted_slices_path, slice_name))
        groundtruth_slice = np.load(os.path.join(groundtruth_slices_path, slice_name))
        mask_slice = np.load(os.path.join(masks_slices_path, slice_name))

        subject_name = slice_name[:slice_name.find("_")]
        slice_index = int(slice_name[slice_name.rfind("_")+1 : slice_name.find(".npy")])

        _mse = mse_2d(test_img=inpainted_slice, ref_img=groundtruth_slice, mask=mask_slice)
        _snr = snr_2d(test_img=inpainted_slice, ref_img=groundtruth_slice, mask=mask_slice)
        _psnr = psnr_2d(test_img=inpainted_slice, ref_img=groundtruth_slice, mask=mask_slice)
        _ssim = ssim_2d(test_img=inpainted_slice, ref_img=groundtruth_slice, mask=mask_slice)

        subject_names.append(subject_name)
        slice_indices.append(slice_index)
        mse_list.append(_mse.item())
        snr_list.append(_snr.item())
        psnr_list.append(_psnr.item())
        ssim_list.append(_ssim.item())

    mse_list = np.array(mse_list)
    snr_list = np.array(snr_list)
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)

    print(f"Dropping {np.sum(np.isnan(mse_list))} NaN values from MSE array")
    pr_mse_list = mse_list[~np.isnan(mse_list)]
    print(f"Dropping {np.sum(np.isnan(snr_list))} NaN values from SNR array")
    pr_snr_list = snr_list[~np.isnan(snr_list)]
    print(f"Dropping {np.sum(np.isnan(psnr_list))} NaN values from PSNR array")
    pr_psnr_list = psnr_list[~np.isnan(psnr_list)]
    print(f"Dropping {np.sum(np.isnan(ssim_list))} NaN values from SSIM array")
    pr_ssim_list = ssim_list[~np.isnan(ssim_list)]

    print("====================================")
    print("Performance Metrics:")
    print(f"MSE: {np.mean(pr_mse_list)} ± {np.std(pr_mse_list)}")
    print(f"SNR: {np.mean(pr_snr_list)} ± {np.std(pr_snr_list)}")
    print(f"PSNR: {np.mean(pr_psnr_list)} ± {np.std(pr_psnr_list)}")
    print(f"SSIM: {np.mean(pr_ssim_list)} ± {np.std(pr_ssim_list)}")
    print("====================================")
    print("Quantiles of Performance Metrics:")
    for quantile in [0.25, 0.5, 0.75]:
        print(f"MSE {quantile}: {np.quantile(pr_mse_list, quantile)}")
        print(f"SNR {quantile}: {np.quantile(pr_snr_list, quantile)}")
        print(f"PSNR {quantile}: {np.quantile(pr_psnr_list, quantile)}")
        print(f"SSIM {quantile}: {np.quantile(pr_ssim_list, quantile)}")
    print("====================================")

    # Save the performance metrics to a Excel file
    performance_metrics = {
        "Subject Name": subject_names,
        "Slice Index": slice_indices,
        "MSE": mse_list,
        "SNR": snr_list,
        "PSNR": psnr_list,
        "SSIM": ssim_list,
    }
    performance_metrics_df = pd.DataFrame(performance_metrics)

    excel_store_dir_path = os.path.join(STORE_EXCEL_BASE_PATH, model_name)
    os.makedirs(excel_store_dir_path, exist_ok=True)
    excel_store_file_path = os.path.join(excel_store_dir_path, f"performance_metrics_{checkpoint_name}_fixed.xlsx")
    performance_metrics_df.to_excel(excel_store_file_path)



def main(models_names: list):
    for model_name in models_names:
        model_inference_path = os.path.join(INFERENCE_SLICES_BASE_PATH, model_name)
        for checkpoint_name in os.listdir(model_inference_path):
            checkpoint_path = os.path.join(model_inference_path, checkpoint_name)
            if not os.path.isdir(checkpoint_path):
                continue

            print(f"Recomputing performance metrics for {model_name} - {checkpoint_name}")
            recompute_metrics_for_checkpoint(model_name, checkpoint_name, checkpoint_path)


if __name__ == "__main__":
    MODELS_NAMES = [
        "ixi_durrer_slicewise_mni",
        "ixi_durrer_slicewise_mni_60_30",
        "ixi_durrer_slicewise_mni_ds228",
        "ixi_durrer_slicewise_mni_s2",

        "ixi_durrer_slicewise_mni_symm_mask",
        "ixi_durrer_slicewise_mni_symm_mask_60_30",
        "ixi_durrer_slicewise_mni_symm_mask_ds228",
        "ixi_durrer_slicewise_mni_symm_mask_s2",

        "ixi_durrer_slicewise_mni_voided",
        "ixi_durrer_slicewise_mni_voided_60_30",
        "ixi_durrer_slicewise_mni_voided_ds228",
        "ixi_durrer_slicewise_mni_voided_s2",
    ]
    main(models_names=MODELS_NAMES)
