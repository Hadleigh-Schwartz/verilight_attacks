import sys
import cv2
import torch
import numpy as np
from hadleigh_utils import pad_image, get_real_mediapipe_results

sys.path.append("facenet-pytorch")
from models.mtcnn import MTCNN


sys.path.append("mediapipe_pytorch/facial_landmarks")
from facial_lm_model import FacialLM_Model 

VIS = True

# prepare models
mtcnn = MTCNN() # face detector
facelandmarker = FacialLM_Model() # facial landmark detector
facelandmarker_weights = torch.load('mediapipe_pytorch/facial_landmarks/model_weights/facial_landmarks.pth')
facelandmarker.load_state_dict(facelandmarker_weights)
facelandmarker = facelandmarker.eval()

img_path = "harry.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.tensor(img, dtype=torch.float32) # emulate format that will be output by generator


# run face detection
bboxes, probs = mtcnn.detect(img_tensor)
bbox = bboxes[0]

x1, y1, x2, y2 = bbox.astype(int)

if VIS:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("image", img[:, :, ::-1]) # for vis only, swap channels for cv2
    cv2.waitKey(0)
    cv2.imshow("cropped_face", img[y1:y2, x1:x2, ::-1])
    cv2.waitKey(0)


cropped_face_tensor = img_tensor[y1:y2, x1:x2, :]

# preprocess face for landmark detection
blob = pad_image(cropped_face_tensor, desired_size=192) # resize
if VIS:
    cv2.imshow("blob", blob.numpy().astype(np.uint8)[:, :, ::-1])
    cv2.waitKey(0)
blob = (blob / 127.5) - 1.0 # normalize
blob = blob.permute(2, 0, 1) # per line 116 of mediapipe_pytorch/facial_landmarks/inference.py, blob is expected to have dimensions [3, H, W]

# run facial landmark detection
facial_landmarks_torch, confidence_torch = facelandmarker.predict(blob) # predict
landmarks_torch = facial_landmarks_torch[0, :, :, :]
landmarks_torch = landmarks_torch.view(468, 3)

if VIS:
    blob_vis = blob.permute(1, 2, 0)
    blob_vis = (blob_vis + 1.0) * 127.5
    blob_vis = blob_vis.numpy().astype(np.uint8)
    blob_vis = np.ascontiguousarray(blob_vis, dtype=np.uint8)
    blob_vis = cv2.cvtColor(blob_vis, cv2.COLOR_RGB2BGR)

    for i in range(landmarks_torch.shape[0]):
        coord = landmarks_torch[i, :]
        x, y = coord[0].item(), coord[1].item()
        cv2.circle(blob_vis, (int(x), int(y)), 1, (0, 255, 0), -1)
        # print(landmarks_np[i])
        # cv2.circle(blob_vis, (int(landmarks_np[i, 0]*blob_vis.shape[1]), int(landmarks_np[i, 1]*blob_vis.shape[2])), 2, (255, 0, 0), -1)
    cv2.imshow("landmarks", blob_vis.astype(np.uint8))
    cv2.waitKey(0)

    get_real_mediapipe_results(img_path)


# run Iris landmark detection, selecting the 5 eye contour landmarks for each eye, for a total of 478 landmarks
# crop blob to left eye
"""
Based off of this resource (https://github.com/Morris88826/MediaPipe_Iris/blob/main/README.md), the iris model expects a 64x64 image of the eye
cropped according to the  (rightEyeUpper0, rightEyeLower0, leftEyeUpper0, leftEyeLower0) landmarks, whose indices are specified 
here: https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
"""
rightEyeUpper0_ids = [246, 161, 160, 159, 158, 157, 173] # landmark indices for the upper eye contour of the right eye
rightEyeLower0_ids = [33, 7, 163, 144, 145, 153, 154, 155, 133]

leftEyeUpper0_ids = [466, 388, 387, 386, 385, 384, 398] # landmark indices for the upper eye contour of the left eye
leftEyeLower0_ids = [263, 249, 390, 373, 374, 380, 381, 382, 362]

rightEyeUpper0 = landmarks_torch[rightEyeUpper0_ids, :]
rightEyeLower0 = landmarks_torch[rightEyeLower0_ids, :]
leftEyeUpper0 = landmarks_torch[leftEyeUpper0_ids, :]
leftEyeLower0 = landmarks_torch[leftEyeLower0_ids, :]

if VIS:
    # draw eye contours on new blob_vis
    blob_vis = blob.permute(1, 2, 0)
    blob_vis = (blob_vis + 1.0) * 127.5
    blob_vis = blob_vis.numpy().astype(np.uint8)
    blob_vis = np.ascontiguousarray(blob_vis, dtype=np.uint8)
    blob_vis = cv2.cvtColor(blob_vis, cv2.COLOR_RGB2BGR)
    for i in range(rightEyeUpper0.shape[0]):
        coord = rightEyeUpper0[i, :]
        x, y = coord[0].item(), coord[1].item()
        cv2.circle(blob_vis, (int(x), int(y)), 1, (0, 255, 0), -1)
    for i in range(rightEyeLower0.shape[0]):
        coord = rightEyeLower0[i, :]
        x, y = coord[0].item(), coord[1].item()
        cv2.circle(blob_vis, (int(x), int(y)), 1, (0, 255, 0), -1)
    for i in range(leftEyeUpper0.shape[0]):
        coord = leftEyeUpper0[i, :]
        x, y = coord[0].item(), coord[1].item()
        cv2.circle(blob_vis, (int(x), int(y)), 1, (0, 255, 0), -1)
    for i in range(leftEyeLower0.shape[0]):
        coord = leftEyeLower0[i, :]
        x, y = coord[0].item(), coord[1].item()
        cv2.circle(blob_vis, (int(x), int(y)), 1, (0, 255, 0), -1)
    cv2.imshow("eye contours", blob_vis.astype(np.uint8))
    cv2.waitKey(0)
