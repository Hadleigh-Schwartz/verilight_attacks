from mp_face_landmarker import PyTorchMediapipeFaceLandmarker
import cv2
from mp_alignment_differentiable import align_landmarks as align_landmarks_differentiable
from mp_alignment_original import align_landmarks as align_landmarks_original
import torch
import numpy as np
from torchviz import make_dot
from hadleigh_utils import compare_to_real_mediapipe, compute_method_differences

import sys
sys.path.append("facenet-pytorch")
from models.mtcnn import MTCNN

def img_demo(img_path):
    mp = PyTorchMediapipeFaceLandmarker()
    # validation on image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img, dtype=torch.float32, requires_grad = True) # emulate format that will be output by generator
    landmarks, blendshapes, padded_face = mp(img_tensor)

    # vis and compare
    padded_face = padded_face.detach().numpy().astype(np.uint8)
    blendshapes_np = blendshapes.detach().numpy()
    landmarks_np = landmarks.detach().numpy()
    compute_method_differences(landmarks_np, blendshapes_np, padded_face)
    compare_to_real_mediapipe(landmarks, blendshapes_np, padded_face, save_landmark_comparison=True)
    W = torch.tensor(padded_face.shape[1])
    H = torch.tensor(padded_face.shape[0])
    aligned3d, aligned2d  = align_landmarks_differentiable(landmarks, W, H, W, H)

    # generate computation graph visualization for mediapipe facemesh 
    # dot = make_dot(landmarks, params=dict(mp.named_parameters()))
    # dot.format = 'png'
    # dot.render('facemesh_graph')



def webcam_demo():
    # webcam live demo
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
        mtcnn = MTCNN()
        bboxes, probs = mtcnn.detect(img)
        if bboxes is None:
            continue
        bbox = bboxes[0]
        bbox = bbox + [-50, -50, 50, 50] # add padding to the bbox, based on observation that mediapipe landmarks extractor benefits from this
        x1, y1, x2, y2 = bbox.astype(int)
        img = img[y1:y2, x1:x2, :]
        img_tensor = torch.tensor(img, dtype=torch.float32, requires_grad = False)
        landmarks, blendshapes, padded_face = mp(img_tensor)
        padded_face = padded_face.detach().numpy().astype(np.uint8)
        blendshapes_np = blendshapes.detach().numpy()
        compare_to_real_mediapipe(landmarks, blendshapes_np, padded_face, live_demo = True)


img_demo("data/obama2.jpg")