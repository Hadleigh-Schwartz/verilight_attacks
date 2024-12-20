"""
Extract and save dynamic features from vidoes in bulk
"""

from dynamic_features_differentiable import VeriLightDynamicFeatures
from hadleigh_utils import aggregate_video_frames
import numpy as np
import os
import glob
import cv2
import torch
import gc

root_dir = "/media/hadleigh/1TB/dynamic_window_level_clips/real_clips"
mod_percs = ["10_20", "20_30", "30_40", "40_50"]
models = ["dagan", "first", "sadtalker", "talklip"]

output_dir = "diff_dynamic_features"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

MIN_FRAMES = 112 # make sure all videos have exactly this number of frames
with torch.no_grad():
    vl = VeriLightDynamicFeatures(device = "cuda", short_range_face_detect=False, long_range_face_detect=True)
    # for mod_perc in mod_percs:
    #     for model in models:
    os.makedirs(f"{output_dir}/real_clips", exist_ok=False)
    videos_paths = glob.glob(f"{root_dir}/*.mp4")
    videos_paths.sort()
    for video_path in videos_paths:
        print("Processing video ", video_path)
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames < MIN_FRAMES:
            cap.release()
            print(f"Skipping video {video_path} as it has {num_frames} total, < {MIN_FRAMES}")
            continue
        cap.release()
        print("Extracting differentiable dynamic features from ", video_path)
        video_name = video_path.split("/")[-1].split(".")[0]
        frames = aggregate_video_frames(video_path, max_frames = 112)
        frames = frames.to(device)
        diff_dynamic_vec = vl(frames)
        diff_dynamic_vec_np = diff_dynamic_vec.detach().cpu().numpy()
        np.save(f"{output_dir}/real_clips/{video_name}_diff_dynamic_vec.npy", diff_dynamic_vec_np)
        gc.collect() # avoid cuda out of memory during repeated inference (https://github.com/ultralytics/ultralytics/issues/4057_

