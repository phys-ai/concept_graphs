import os
import pandas as pd
import json
from PIL import Image
import shutil

# Load DataFrames
attr_df = pd.read_csv("input/celeba-dataset/list_attr_celeba.csv")
bbox_df = pd.read_csv("input/celeba-dataset/list_bbox_celeba.csv").set_index('image_id')

# Setup experiment parameters
ntrain_sample = 10000
ntest_sample = 100
out_dir = f"working/celeba-3classes-smiling-{ntrain_sample}_{ntest_sample}/"

# Ensure necessary directories exist
for subdir in ["", "train/", "test/"]:
    os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)

# Filter attributes
attr_df = attr_df[["image_id", "Male", "Smiling", "Black_Hair", "Blond_Hair"]]

# Load experiment configurations
with open("config_category.json", 'r') as f:
    configs = json.load(f)
test_configs = configs["H32-train1"]["test"]

# Process and copy images based on conditions
counts = {}
for Male, Smiling, Hair_Color in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]:
    cond = (attr_df["Male"] == Male) & (attr_df["Smiling"] == Smiling)
    cond &= (attr_df["Black_Hair"] == 1) if Hair_Color == 1 else (attr_df["Blond_Hair"] == 1)
    label = f"{int(Male == 1)}{int(Smiling == -1)}{int(Hair_Color == -1)}"
    image_files = attr_df[cond]["image_id"]
    nmax = ntest_sample if label in test_configs else ntrain_sample

    print(label, nmax, len(image_files))
    for idx, image_file in enumerate(image_files):
        source_file = f"input/celeba-dataset/img_align_celeba/img_align_celeba/{image_file}"
        dest_subdir = "test/" if idx < nmax else "train/"
        destination_file = f"{out_dir}{dest_subdir}celeba_{label}_{image_file.split('/')[-1]}"
        shutil.copy2(source_file, destination_file)



