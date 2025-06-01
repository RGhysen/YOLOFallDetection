import random
from torch.utils.data import DataLoader
from dataloader import get_video_prefixes, FullVideoDataset, collate_fn, data_folder

# Discover all video prefixes
all_vids = get_video_prefixes(data_folder)

# Shuffle with fixed seed for reproducibility
random.seed(39)
random.shuffle(all_vids)

# Define proportions for train/val/test
train_pct = 0.9
val_pct   = 0.0
# test_pct will be whatever remains (0.1)

n_total   = len(all_vids)
n_train   = int(train_pct * n_total)
n_val     = int(val_pct   * n_total)

# Slice into three disjoint sets
train_vids = all_vids[:n_train]                   
val_vids   = all_vids[n_train : n_train + n_val]  
test_vids  = all_vids[n_train + n_val :]    

# Quick‐run subset for testing purposes
subset_frac = 1
train_vids = train_vids[: max(1, int(subset_frac * len(train_vids)))]
val_vids   = val_vids[:   max(1, int(subset_frac * len(val_vids)))]
test_vids  = test_vids[:  max(1, int(subset_frac * len(test_vids)))]

print("Train videos:", train_vids)
print(" Val videos:", val_vids)
print("Test videos:", test_vids)

# Build datasets for each split
train_ds = FullVideoDataset(data_folder, train_vids, is_train=True)
val_ds   = FullVideoDataset(data_folder, val_vids,   is_train=False)
test_ds  = FullVideoDataset(data_folder, test_vids,  is_train=False)

# Create DataLoaders; shuffle only the training split at the video level
train_loader = DataLoader(
    train_ds,
    batch_size=1, # videos per batch
    shuffle=True,         
    num_workers=4,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,         
    num_workers=0,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,         
    num_workers=0,
    collate_fn=collate_fn
)