import torch
from huggingface_hub import hf_hub_download
import cv2
import numpy as np

device = 'cpu'

# Load model and weights
landmark_model_file = hf_hub_download(repo_id='py-feat/mp_facemesh_v2', filename="face_landmarks_detector_Nx3x256x256_onnx.pth")
landmark_detector = torch.load(landmark_model_file, map_location=device, weights_only=False)
landmark_detector.eval()
landmark_detector.to(device)


# Test model
face_image = "obama2.jpeg"  # Replace with your extracted face image that is [224, 224]
face_image = cv2.imread(face_image)
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
# resize to 224x224
face_image = cv2.resize(face_image, (224, 224))
face_image = torch.tensor(face_image, dtype=torch.float32)
face_image = face_image.permute(2, 0, 1)

# Extract Landmarks
landmark_results = landmark_detector(face_image.to(device))
        