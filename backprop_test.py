


# read in a video
# create a dummy vector of same dimensionality of our dynamic one
# pass video htyrough the model, which just returns the frames exactly
# loss function compute dynamic feature vec 
# compare with dummy

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
from dynamic_features_differentiable import VeriLightDynamicFeatures
import torch
import cv2
from torchviz import make_dot
from hadleigh_utils import aggregate_video_frames
import matplotlib.pyplot as plt


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
 
vl = VeriLightDynamicFeatures(long_range_face_detect=False, short_range_face_detect=False)
def vl_loss(x, vec):
    dynamic_feature_vec = vl(frames)
    loss = torch.nn.MSELoss()(dynamic_feature_vec, vec)
    return loss

torch.autograd.set_detect_anomaly(True)
frames = aggregate_video_frames("data/test_video1.mp4", 90)
loss = vl_loss(frames, torch.zeros(1440))
loss.backward()
print(loss)

# optimizer.zero_grad()
# loss.backward()
# optimizer.step()



