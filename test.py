import sys
import cv2
import torch
import numpy as np
from hadleigh_utils import pad_image, get_real_mediapipe_results

sys.path.append("facenet-pytorch")
from models.mtcnn import MTCNN


sys.path.append("mediapipe_pytorch/facial_landmarks")
from facial_lm_model import FacialLM_Model 

sys.path.append("mediapipe_pytorch/iris")
from irismodel import IrisLM


VIS = True

# prepare models
mtcnn = MTCNN() # face detector

# MediaPipe facial landmark detector
facelandmarker = FacialLM_Model() 
facelandmarker_weights = torch.load('mediapipe_pytorch/facial_landmarks/model_weights/facial_landmarks.pth')
facelandmarker.load_state_dict(facelandmarker_weights)
facelandmarker = facelandmarker.eval()

# MediaPipe iris landmark detector
irislandmarker = IrisLM()
irislandmarker_weights = torch.load('mediapipe_pytorch/iris/model_weights/irislandmarks.pth')
irislandmarker.load_state_dict(irislandmarker_weights)
irislandmarker = irislandmarker.eval()



# load image
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

"""
Based off of this resource (https://github.com/Morris88826/MediaPipe_Iris/blob/main/README.md), the iris model expects a 64x64 image of the eye
cropped according to the  (rightEyeUpper0, rightEyeLower0, leftEyeUpper0, leftEyeLower0) landmarks, whose indices are specified 
here: https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts

It is also mentioned that a significant (0.25 or 1, the README is ambiguous), is added to the crop of the eye.
"""

# get regions of interest for each eye
rightEyeUpper0_ids = [246, 161, 160, 159, 158, 157, 173] # landmark indices for the upper eye contour of the right eye
rightEyeLower0_ids = [33, 7, 163, 144, 145, 153, 154, 155, 133]

leftEyeUpper0_ids = [466, 388, 387, 386, 385, 384, 398] # landmark indices for the upper eye contour of the left eye
leftEyeLower0_ids = [263, 249, 390, 373, 374, 380, 381, 382, 362]

rightEyeUpper0 = landmarks_torch[rightEyeUpper0_ids, :]
rightEyeLower0 = landmarks_torch[rightEyeLower0_ids, :]
leftEyeUpper0 = landmarks_torch[leftEyeUpper0_ids, :]
leftEyeLower0 = landmarks_torch[leftEyeLower0_ids, :]

right_eye_left = rightEyeUpper0[:, 0].min().item()
right_eye_right = rightEyeUpper0[:, 0].max().item()
right_eye_width = right_eye_right - right_eye_left
horizontal_margin = 0.5 * right_eye_width
right_eye_left -= horizontal_margin
right_eye_right += horizontal_margin

right_eye_top = rightEyeUpper0[:, 1].min().item()
right_eye_bottom = rightEyeLower0[:, 1].max().item()
right_eye_height = right_eye_bottom - right_eye_top
vertical_margin = 0.5 * right_eye_height
right_eye_top -= vertical_margin
right_eye_bottom += vertical_margin

left_eye_left = leftEyeUpper0[:, 0].min().item()
left_eye_right = leftEyeUpper0[:, 0].max().item()
left_eye_width = left_eye_right - left_eye_left
left_horizontal_margin = 0.5 * left_eye_width
left_eye_left -= left_horizontal_margin
left_eye_right += left_horizontal_margin

left_eye_top = leftEyeUpper0[:, 1].min().item()
left_eye_bottom = leftEyeLower0[:, 1].max().item()
left_eye_height = left_eye_bottom - left_eye_top
left_vertical_margin = 0.5 * left_eye_height
left_eye_top -= left_vertical_margin
left_eye_bottom += left_vertical_margin

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

    cv2.rectangle(blob_vis, (int(right_eye_left), int(right_eye_top)), (int(right_eye_right), int(right_eye_bottom)), (0, 255, 0), 2)
    cv2.rectangle(blob_vis, (int(left_eye_left), int(left_eye_top)), (int(left_eye_right), int(left_eye_bottom)), (0, 255, 0), 2)

    cv2.imshow("eye contours", blob_vis.astype(np.uint8))
    cv2.waitKey(0)

blob_unperm = blob.permute(1, 2, 0)
blob_unnorm = (blob_unperm + 1.0) * 127.5
left_eye_crop = blob_unnorm [int(left_eye_top):int(left_eye_bottom), int(left_eye_left):int(left_eye_right), :]
right_eye_crop = blob_unnorm[int(right_eye_top):int(right_eye_bottom), int(right_eye_left):int(right_eye_right), :]

# pad the crops
left_eye_crop = pad_image(left_eye_crop, desired_size=64)
right_eye_crop = pad_image(right_eye_crop, desired_size=64)

if VIS:
    left_eye_crop_vis = left_eye_crop.numpy().astype(np.uint8)
    left_eye_crop_vis = np.ascontiguousarray(left_eye_crop_vis, dtype=np.uint8)
    left_eye_crop_vis = cv2.cvtColor(left_eye_crop_vis, cv2.COLOR_RGB2BGR)
    cv2.imshow("left_eye_crop", left_eye_crop_vis)
    cv2.waitKey(0)

    right_eye_crop_vis = right_eye_crop.numpy().astype(np.uint8)
    right_eye_crop_vis = np.ascontiguousarray(right_eye_crop_vis, dtype=np.uint8)
    right_eye_crop_vis = cv2.cvtColor(right_eye_crop_vis, cv2.COLOR_RGB2BGR)
    cv2.imshow("right_eye_crop", right_eye_crop_vis)
    cv2.waitKey(0)


# in mediapipe_torch/iris/inference.py, the input image is expected to be normalized be 255, differently than  ... 1.0) * 127.5 business above
left_eye_crop /= 255
right_eye_crop /= 255
left_eye_crop = left_eye_crop.permute(2, 0, 1)
right_eye_crop = right_eye_crop.permute(2, 0, 1)

# run iris detection
left_eye_contour_torch, left_iris_torch = irislandmarker.predict(left_eye_crop)
right_eye_contour_torch, right_iris_torch = irislandmarker.predict(right_eye_crop)

left_iris_landmarks_torch = left_iris_torch.view(5, 3)

right_iris_landmarks_torch = right_iris_torch.view(5, 3)

if VIS:
    for i in range(left_iris_landmarks_torch.shape[0]):
        coord = left_iris_landmarks_torch[i, :]
        x, y = coord[0].item(), coord[1].item()
        cv2.circle(left_eye_crop_vis, (int(x), int(y)), 1, (0, 255, 0), -1)
    cv2.imshow("left_iris_landmarks", left_eye_crop_vis.astype(np.uint8))
    cv2.waitKey(0)

    for i in range(right_iris_landmarks_torch.shape[0]):
        coord = right_iris_landmarks_torch[i, :]
        x, y = coord[0].item(), coord[1].item()
        cv2.circle(right_eye_crop_vis, (int(x), int(y)), 1, (0, 255, 0), -1)
    cv2.imshow("right_iris_landmarks", right_eye_crop_vis.astype(np.uint8)) 
    cv2.waitKey(0)



# now adjust the iris landmarks to the original image pixel space
left_eye_ratio = float(64)/max(left_eye_height*2, left_eye_width*2)
new_left_eye_size = tuple([int(x*left_eye_ratio) for x in [left_eye_height*2, left_eye_width*2]])
left_eye_delta_w = 64 - new_left_eye_size [1]
left_eye_delta_h = 64 - new_left_eye_size [0]
left_eye_top_padding, left_eye_bottom_padding = left_eye_delta_h//2, left_eye_delta_h-(left_eye_delta_h//2)
left_eye_left_padding, left_eye_right_padding = left_eye_delta_w//2, left_eye_delta_w-(left_eye_delta_w//2)

left_iris_landmarks_torch[:, 0] = left_iris_landmarks_torch[:, 0] - left_eye_left_padding
left_iris_landmarks_torch[:, 0] = left_iris_landmarks_torch[:, 0]*(left_eye_width*2/new_left_eye_size[1]) + left_eye_left
left_iris_landmarks_torch[:, 1] = left_iris_landmarks_torch[:, 1] - left_eye_top_padding
left_iris_landmarks_torch[:, 1] = left_iris_landmarks_torch[:, 1]*(left_eye_height*2/new_left_eye_size[0]) + left_eye_top

right_eye_ratio = float(64)/max(right_eye_height*2, right_eye_width*2)
new_right_eye_size = tuple([int(x*right_eye_ratio) for x in [right_eye_height*2, right_eye_width*2]])
right_eye_delta_w = 64 - new_right_eye_size [1]
right_eye_delta_h = 64 - new_right_eye_size [0]
right_eye_top_padding, right_eye_bottom_padding = right_eye_delta_h//2, right_eye_delta_h-(right_eye_delta_h//2)
right_eye_left_padding, right_eye_right_padding = right_eye_delta_w//2, right_eye_delta_w-(right_eye_delta_w//2)


right_iris_landmarks_torch[:, 0] = right_iris_landmarks_torch[:, 0] - right_eye_left_padding
right_iris_landmarks_torch[:, 0] = right_iris_landmarks_torch[:, 0]*(right_eye_width*2/new_right_eye_size[1]) + right_eye_left
right_iris_landmarks_torch[:, 1] = right_iris_landmarks_torch[:, 1] - right_eye_top_padding
right_iris_landmarks_torch[:, 1] = right_iris_landmarks_torch[:, 1]*(right_eye_height*2/new_right_eye_size[0]) + right_eye_top

if VIS:
    for i in range(left_iris_landmarks_torch.shape[0]):
        coord = left_iris_landmarks_torch[i, :]
        x, y = coord[0].item(), coord[1].item()
        cv2.circle(blob_vis, (int(x), int(y)), 1, (0, 0, 255), -1)
    for i in range(right_iris_landmarks_torch.shape[0]):
        coord = right_iris_landmarks_torch[i, :]
        x, y = coord[0].item(), coord[1].item()
        cv2.circle(blob_vis, (int(x), int(y)), 1, (0, 0, 255), -1)
    cv2.imshow("iris_landmarks", blob_vis.astype(np.uint8))
    cv2.waitKey(0)



