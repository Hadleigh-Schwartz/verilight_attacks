
from dynamic_features_differentiable import VeriLightDynamicFeatures
import torch
import cv2
from torchviz import make_dot
from hadleigh_utils import aggregate_video_frames
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

frames = aggregate_video_frames("/media/hadleigh/1TB/dynamic_window_level_clips/real_clips/charlie_0_60_close_0.mp4", 90)
frames = frames.to("cuda")
vl = VeriLightDynamicFeatures(long_range_face_detect=True, short_range_face_detect=False, device = "cuda")
dynamic_feature_vec = vl(frames)

dynamic_feature_vec_np = dynamic_feature_vec.detach().numpy()
np.save("charlie_0_60_close_0_diff_dynamic_vec.npy", dynamic_feature_vec_np)
plt.plot(dynamic_feature_vec_np)
plt.savefig("dynamic_feature_vec.png")

# vec1 = np.load("charlie_0_front_close_0_diff_dynamic_vec.npy")
# vec2 = np.load("charlie_0_45_close_0_diff_dynamic_vec.npy")
# vec3 = np.load("charlie_0_60_close_0_diff_dynamic_vec.npy")

# corr12, _  = scipy.stats.pearsonr(vec1, vec2)
# corr13, _ = scipy.stats.pearsonr(vec1, vec3)
# corr23, _ = scipy.stats.pearsonr(vec2, vec3)
# plt.plot(vec1, label = "charlie_0_front_close_0")
# plt.plot(vec2, label = "charlie_0_45_close_0")
# plt.plot(vec3, label = "charlie_0_60_close_0")
# plt.legend()
# plt.title(f"Pearson correlation: {corr12}, {corr13}, {corr23}")
# plt.savefig("charlie_0_diff_dynamic_vec.png")

# # save as np
# # np.save("test_dynamic_feature_vec.npy", dynamic_feature_vec.detach().numpy())

# # generate computation graph visualization for verilight dynamic feature extractror
# # sys.setrecursionlimit(10000) # to avoid RecursionError when building the graph
# # dot = make_dot(test, params=dict(vl.named_parameters()))
# # dot.format = 'png'
# # dot.render('dyn_feat_graph')

