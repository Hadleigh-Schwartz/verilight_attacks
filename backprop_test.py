import torch
import torch.optim as optim
from dynamic_features_differentiable import VeriLightDynamicFeatures
from hadleigh_utils import aggregate_video_frames
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)
frames = aggregate_video_frames("data/test_video1.mp4", 90)
model = VeriLightDynamicFeatures(long_range_face_detect=False, short_range_face_detect=False)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
pred_vec = model(frames)
loss = torch.nn.MSELoss()(pred_vec, torch.zeros_like(pred_vec))
optimizer.zero_grad()
loss.backward()
optimizer.step()
# Print gradients for each parameter in your model
for name, param in model.named_parameters():
    print(name, param.grad)


