"""
Extract and save dynamic features from vidoes in bulk
"""

from dynamic_features_differentiable import VeriLightDynamicFeatures
from hadleigh_utils import aggregate_video_frames
import numpy as np

video_path = "data/test_video1.mp4"
video_name = video_path.split("/")[-1].split(".")[0]

frames = aggregate_video_frames(video_path)
vl = VeriLightDynamicFeatures()
diff_dynamic_vec = vl(frames)
diff_dynamic_vec_np = diff_dynamic_vec.detach().numpy()
np.save(f"{video_name}_diff_dynamicvec.npy", diff_dynamic_vec_np)
