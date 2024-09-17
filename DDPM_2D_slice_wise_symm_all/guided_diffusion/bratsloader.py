import math
import os
import os.path

import nibabel
import nibabel as nib
import numpy as np
import torch
import torch.nn


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True, override_seqtypes=None, ref_mask="mask"):
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag = test_flag
        if test_flag:
            # Originally: self.seqtypes = ["voided", "mask"]
            self.seqtypes = override_seqtypes or ["healthy-voided", "mask", "symm-healthy-mask", "symm-t1n", "t1n"]
        else:
            # Originally: self.seqtypes = ["diseased", "mask", "healthy"]
            self.seqtypes = override_seqtypes or ["healthy-voided", "mask", "symm-healthy-mask", "symm-t1n", "t1n"]

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        self.mask_vis = []
        for root, dirs, files in os.walk(self.directory):
            dirs_sorted = sorted(dirs)
            for dir_id in dirs_sorted:
                datapoint = dict()
                sli_dict = dict()
                for ro, di, fi in os.walk(root + "/" + str(dir_id)):
                    fi_sorted = sorted(fi)
                    # print(fi_sorted)
                    for f in fi_sorted:
                        # Original: seqtype = f.split("-")[-1].split(".")[0]
                        seqtype = f[f.find(dir_id)+len(dir_id)+1:f.rfind(".nii.gz")]
                        datapoint[seqtype] = os.path.join(root, dir_id, f)
                        ############################################################
                        if "BraTS-GLI-0166" in f and not self.test_flag:  # NEW
                            print(f"Ignoring {f} to use in Validation ...")
                            continue
                        ############################################################
                        if seqtype == ref_mask:
                            slice_range = []
                            mask_to_define_rand = np.array(
                                nibabel.load(datapoint[ref_mask]).dataobj
                            )
                            for i in range(0, 224):
                                mask_slice = mask_to_define_rand[:, :, i]
                                if np.sum(mask_slice) != 0:
                                    slice_range.append(i)

                    if not self.seqtypes_set.issubset(set(datapoint.keys())):
                        raise AssertionError(f"""
                            Datapoint is incomplete.\n
                            Datapoint keys are {datapoint.keys()}\n
                            Expected keys are {self.seqtypes}
                        """)
                    self.database.append(datapoint)
                    self.mask_vis.append(slice_range)

            break

    def __getitem__(self, x):
        filedict = self.database[x]
        slicedict = self.mask_vis[x]

        out_single = []

        if self.test_flag:
            for seqtype in self.seqtypes:
                nib_img = np.array(nibabel.load(filedict[seqtype]).dataobj).astype(np.float32)
                path = filedict[seqtype]
                img_preprocessed = torch.tensor(nib_img)
                out_single.append(img_preprocessed)

            out_single = torch.stack(out_single)
            image = out_single[0:5, ...]  # voided, binary mask, symm mask, symm t1n, t1n
            path = filedict[seqtype]

            return (image, path, slicedict)

        else:
            for seqtype in self.seqtypes:
                nib_img = np.array(nibabel.load(filedict[seqtype]).dataobj).astype(np.float32)
                path = filedict[seqtype]
                img_preprocessed = torch.tensor(nib_img)
                out_single.append(img_preprocessed)

            out_single = torch.stack(out_single)

            image = out_single[0:4, ...]  # healthy, mask, symm mask, symm t1n
            label = out_single[4, ...]  # t1n
            label = label.unsqueeze(0)
            path = filedict[seqtype]

            # Originally: return (image, label, path, slicedict)
            return (image, label, slicedict)

    def __len__(self):
        return len(self.database)
