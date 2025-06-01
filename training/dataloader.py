import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
import torch
import numpy as np

ORIGINAL_FPS = 30
TARGET_FPS   = 7 # Set to 7 fps because limit of RBPI inference hardware
SAMPLE_STRIDE = int(round(ORIGINAL_FPS / TARGET_FPS))

def get_video_prefixes(data_folder, sep="_"):
    """
    Scan data_folder/images for all files, extract the part before the first seperator
    as the video prefix. "angle3-1_frame0001.jpg" → "angle3-1".
    Returns a sorted list of unique prefixes.
    """
    images_dir = os.path.join(data_folder, "images")
    # Grab every image file under images_dir
    all_paths = glob.glob(os.path.join(images_dir, "*.*"))
    prefixes = set()
    for p in all_paths:
        name   = os.path.basename(p)
        prefix = name.split(sep, 1)[0]
        prefixes.add(prefix)
    return sorted(prefixes)


class FullVideoDataset(Dataset):
    """
    A PyTorch Dataset that returns entire videos as sequences of frames,
    plus YOLO-formatted ground-truth boxes. Applies one video-level augmentation
    via ReplayCompose so all frames share identical transformations.
    """
    def __init__(self,
                 data_folder: str,
                 video_prefixes: list,
                 img_size: tuple=(320,320),
                 is_train: bool=True):
        super().__init__()
        self.img_size   = img_size
        self.is_train   = is_train

        # Build a dict: { prefix -> [list of frame paths] }
        images_dir = os.path.join(data_folder, "images")
        all_imgs   = sorted(glob.glob(os.path.join(images_dir, "*.*")))
        self.video_dict = {}
        for p in all_imgs:
            prefix = os.path.basename(p).split("_",1)[0]
            if video_prefixes and prefix not in video_prefixes:
                continue
            self.video_dict.setdefault(prefix, []).append(p)

        # Ensure chronological order within each video
        for prefix in self.video_dict:
            self.video_dict[prefix].sort()

        # One item per video prefix
        self.indices = list(self.video_dict.keys())

        # Set up per-video augmentations
        if is_train:
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Resize(*img_size)
            ]
        else:
            transforms = [ A.Resize(*img_size) ]

        # ReplayCompose will store the random params on first call,
        # then allow .replay() to apply the same to all frames.
        self.aug = A.ReplayCompose(
            transforms,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["category_ids"]
            )
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Identify which video -> list of frame paths
        prefix      = self.indices[idx]
        frame_paths = self.video_dict[prefix][::SAMPLE_STRIDE]

        # Load all raw frames & their YOLO-normalized labels
        imgs       = []
        all_bboxes = []
        all_labels = []
        for p in frame_paths:
            # load image
            arr_img = np.array(Image.open(p).convert("RGB"))
            imgs.append(arr_img)

            # corresponding label .txt
            lbl_path = os.path.join(
                os.path.dirname(p).replace("images","labels"),
                os.path.basename(p).rsplit(".",1)[0] + ".txt"
            )
            if os.path.exists(lbl_path):
                arr = np.loadtxt(lbl_path)
                # ensure shape [N,5]
                if arr.ndim == 1:
                    arr = arr[None,:]
                all_labels.append(arr[:,0].astype(int).tolist())
                all_bboxes.append(arr[:,1:].tolist())
            else:
                # no GT -> empty
                all_labels.append([])
                all_bboxes.append([])

        # Generate one random augmentation based on the first frame
        first = self.aug(
            image=imgs[0],
            bboxes=all_bboxes[0],
            category_ids=all_labels[0]
        )
        replay = first["replay"]

        # Apply the same transform to every subsequent frame
        aug_imgs   = [ first["image"] ]
        aug_boxes  = [ first["bboxes"] ]
        aug_labels = [ first["category_ids"] ]
        for img, bboxes, labels in zip(imgs[1:], all_bboxes[1:], all_labels[1:]):
            out = A.ReplayCompose.replay(
                replay,
                image=img,
                bboxes=bboxes,
                category_ids=labels
            )
            aug_imgs.append(out["image"])
            aug_boxes.append(out["bboxes"])
            aug_labels.append(out["category_ids"])

        # Convert images -> Tensor [T, C, H, W]
        tensor_imgs = torch.stack([
            torch.tensor(im, dtype=torch.float32).permute(2,0,1)
            for im in aug_imgs
        ], dim=0)

        # Convert YOLO-norm bboxes -> absolute [x1,y1,x2,y2] per frame
        targets = []
        H, W = self.img_size
        for bboxes, labels in zip(aug_boxes, aug_labels):
            if bboxes:
                xyxy = []
                for (xc,yc,w_box,h_box) in bboxes:
                    x1 = (xc - w_box/2) * W
                    y1 = (yc - h_box/2) * H
                    x2 = (xc + w_box/2) * W
                    y2 = (yc + h_box/2) * H
                    xyxy.append([x1,y1,x2,y2])
                boxes_tensor  = torch.tensor(xyxy, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.long)
            else:
                boxes_tensor  = torch.empty((0,4), dtype=torch.float32)
                labels_tensor = torch.empty((0,),   dtype=torch.long)

            targets.append({
                "boxes":  boxes_tensor,
                "labels": labels_tensor
            })

        # Return one video worth of frames + targets
        return tensor_imgs, targets


def collate_fn(batch):
    """
    Custom collate for variable-length videos:
      Input `batch` is a list of (tensor_imgs, targets) tuples.
      We return a dict with two lists so the training loop can iterate
      per-video naturally.
    """
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return {"images": images, "targets": targets}

# --- Example usage: discover & split videos ---
data_folder = "/data/own_data/full_dataset"
all_vids    = get_video_prefixes(data_folder)