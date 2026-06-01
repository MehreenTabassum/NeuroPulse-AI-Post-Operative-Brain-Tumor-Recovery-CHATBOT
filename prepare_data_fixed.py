import os
import shutil
import random
import nibabel as nib
from pathlib import Path
import numpy as np

# Paths
RAW_DIR   = "./UCSD-PTGBM-BraTS-2024-test-set"
OUT_DIR   = "./data"
VAL_SPLIT = 0.2   # 80/20 train/val
SEED      = 42

random.seed(SEED)

timepoints = sorted(Path(RAW_DIR).glob("UCSD-PTGBM-*"))
print(f"Found {len(timepoints)} timepoint directories inside '{RAW_DIR}'...")

if len(timepoints) == 0:
    print("❌ Error: No folders were discovered. Check RAW_DIR path.")
    exit(1)

# First Pass: Gather all valid cases and calculate their voxel volumes
valid_cases = []
voxel_counts = []

print("Analyzing scan volumes to calculate dynamic threshold...")
for tp in timepoints:
    seg_files = list(tp.glob("*_BraTS_tumor_seg.nii.gz"))
    t1c_files = list(tp.glob("*_T1post.nii.gz"))
    
    if seg_files and t1c_files:
        seg_file = seg_files[0]
        t1c_file = t1c_files[0]

        # Load the segmentation mask
        seg = nib.load(str(seg_file)).get_fdata()
        # Handle multi-class vs binary masks
        et_voxels = (seg == 3).sum() if seg.max() >= 3 else (seg > 0).sum()
        
        valid_cases.append((t1c_file, et_voxels))
        voxel_counts.append(et_voxels)

if not valid_cases:
    print("❌ Error: No valid scans with matching segments and T1 sequences found.")
    exit(1)

# Calculate Median to enforce a perfect 50/50 balanced dataset split
median_threshold = np.median(voxel_counts)
print(f"\n📊 Dataset Stats:")
print(f"   • Total Scans Found : {len(valid_cases)}")
print(f"   • Smallest Tumor    : {min(voxel_counts)} voxels")
print(f"   • Largest Tumor     : {max(voxel_counts)} voxels")
print(f"   • Calculated Median : {median_threshold} voxels\n")

# Second Pass: Distribute into train/val and class_0/class_1 based on median
copied_count = 0

for t1c_file, et_voxels in valid_cases:
    # Split strictly relative to the median of your cohort
    label = "class_1" if et_voxels >= median_threshold else "class_0"
    split = "val" if random.random() < VAL_SPLIT else "train"

    dest = Path(OUT_DIR) / split / label
    dest.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(str(t1c_file), dest / t1c_file.name)
    copied_count += 1

print(f"✔ Dataset organization complete! Balanced {copied_count} scans into '{OUT_DIR}'.")