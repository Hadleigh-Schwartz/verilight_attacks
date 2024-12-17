
from dynamic_features_differentiable import VeriLightDynamicFeatures
import torch
import cv2
from torchviz import make_dot
from hadleigh_utils import aggregate_video_frames
import matplotlib.pyplot as plt

frames = aggregate_video_frames("data/test_video1.mp4", 300)
vl = VeriLightDynamicFeatures(long_range_face_detect=False, short_range_face_detect=True)
dynamic_feature_vec = vl(frames)

dynamic_feature_vec_np = dynamic_feature_vec.detach().numpy()
plt.plot(dynamic_feature_vec_np)
plt.savefig("dynamic_feature_vec.png")

# save as np
# np.save("test_dynamic_feature_vec.npy", dynamic_feature_vec.detach().numpy())

# generate computation graph visualization for verilight dynamic feature extractror
# sys.setrecursionlimit(10000) # to avoid RecursionError when building the graph
# dot = make_dot(test, params=dict(vl.named_parameters()))
# dot.format = 'png'
# dot.render('dyn_feat_graph')
