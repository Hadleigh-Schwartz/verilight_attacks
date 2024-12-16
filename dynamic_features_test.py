
from dynamic_features_differentiable import VeriLightDynamicFeatures
import torch
import cv2
from torchviz import make_dot
from hadleigh_utils import aggregate_video_frames

frames = aggregate_video_frames("data/test_video1.mp4")
vl = VeriLightDynamicFeatures()
dynamic_feature_vec = vl(frames)

# save as np
# np.save("test_dynamic_feature_vec.npy", dynamic_feature_vec.detach().numpy())

# generate computation graph visualization for verilight dynamic feature extractror
# sys.setrecursionlimit(10000) # to avoid RecursionError when building the graph
# dot = make_dot(test, params=dict(vl.named_parameters()))
# dot.format = 'png'
# dot.render('dyn_feat_graph')

