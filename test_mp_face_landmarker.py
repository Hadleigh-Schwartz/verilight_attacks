from mp_face_landmarker import PyTorchMediapipeFaceLandmarker
import cv2
import torch
import numpy as np
from torchviz import make_dot
from hadleigh_utils import compare_to_real_mediapipe, compute_method_differences

import sys
sys.path.append("facenet-pytorch")
from models.mtcnn import MTCNN

def img_demo(img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mp = PyTorchMediapipeFaceLandmarker(device, long_range_face_detect=True, short_range_face_detect=False).to(device)
    # validation on image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img, dtype=torch.float32, requires_grad = True).to(device) # emulate format that will be output by generator
    landmarks, blendshapes, padded_face, bbox = mp(img_tensor)

    # vis and compare
    padded_face = padded_face.detach().cpu().numpy().astype(np.uint8)
    blendshapes_np = blendshapes.detach().cpu().numpy()
    landmarks_np = landmarks.detach().cpu().numpy()
    compare_to_real_mediapipe(landmarks, blendshapes_np, padded_face, save_landmark_comparison=True, display=False, save=True)


    # generate computation graph visualization for mediapipe facemesh 
    # dot = make_dot(landmarks, params=dict(mp.named_parameters()))
    # dot.format = 'png'
    # dot.render('facemesh_graph')



def webcam_demo():
    # webcam live demo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mp = PyTorchMediapipeFaceLandmarker(device).to(device)
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        count += 1
        if count < 4:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # mtcnn = MTCNN()
        # bboxes, probs = mtcnn.detect(img)
        # if bboxes is None:
        #     continue
        # bbox = bboxes[0]
        # bbox = bbox + [-50, -50, 50, 50] # add padding to the bbox, based on observation that mediapipe landmarks extractor benefits from this
        # x1, y1, x2, y2 = bbox.astype(int)
        # img = img[y1:y2, x1:x2, :]
        img_tensor = torch.tensor(img, dtype=torch.float32, requires_grad = False).to(device) # emulate format that will be output by generator
        landmarks, blendshapes, padded_face, bbox = mp(img_tensor)
        padded_face = padded_face.detach().numpy().astype(np.uint8)
        blendshapes_np = blendshapes.detach().numpy()
        compare_to_real_mediapipe(landmarks, blendshapes_np, padded_face, live_demo = True)

webcam_demo()