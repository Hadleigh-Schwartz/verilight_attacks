"""
Extract and save dynamic features from vidoes in bulk
"""

from dynamic_features_differentiable import VeriLightDynamicFeatures
from hadleigh_utils import aggregate_video_frames
import numpy as np
import os
import glob
import cv2

# assuming fps of 26 for these vids -> 117 frames

root_dir = "/media/lex/1TB/dynamic_window_level_clips/mod_clips_adjusted/"
mod_percs = ["10_20", "20_30", "30_40", "40_50"]
models = ["dagan", "first", "sadtalker", "talklip"]

output_dir = "diff_dynamic_features"
vl = VeriLightDynamicFeatures()
for mod_perc in mod_percs:
    for model in models:
        os.makedirs(f"{output_dir}/{mod_perc}/{model}", exist_ok=False)
        videos_paths = glob.glob(f"{root_dir}/{mod_perc}/{model}/*.mp4")
        for video_path in videos_paths:
            print("Extracting differentiable dynamic features from ", video_path)
            # get number of frames in the video
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count < 117:
                print("Skipping video ", video_path, " as it has less than 90 frames")
                continue
            video_name = video_path.split("/")[-1].split(".")[0]
            frames = aggregate_video_frames(video_path, 117)
            diff_dynamic_vec = vl(frames)
            diff_dynamic_vec_np = diff_dynamic_vec.detach().numpy()
            np.save(f"{output_dir}/{mod_perc}/{model}/{video_name}_diff_dynamic_vec.npy", diff_dynamic_vec_np)
