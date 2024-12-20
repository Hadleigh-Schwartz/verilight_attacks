import torch
import torch.optim as optim
from dynamic_features_differentiable import VeriLightDynamicFeatures
from hadleigh_utils import aggregate_video_frames
import time

torch.autograd.set_detect_anomaly(True)

frames1 = aggregate_video_frames("data/test_video1.mp4", 90) # read in video as 4D tensor
# frames2 = aggregate_video_frames("data/test_video2.mp4", 90) # read in video as 4D tensor
# frames_batch = torch.stack([frames1, frames2])

# initialize opt and my model
device = "cuda"
frames1 = frames1.to(device)
model = VeriLightDynamicFeatures(device, long_range_face_detect=False, short_range_face_detect=False)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)       

n_epochs = 50
for epoch in range(n_epochs):
    # run a forward pass and backward pass
    start_pred = time.time()
    pred_vec = model(frames1)
    end_pred = time.time()
    loss = torch.nn.MSELoss()(pred_vec, 
    torch.zeros_like(pred_vec))
    optimizer.zero_grad()
    loss.backward()
    loss_val = loss.item()
    optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, loss_val) + ". Inference time (s): " + str(end_pred - start_pred))


# # print gradients
# for name, param in model.named_parameters():
#     print(name, param.grad)


