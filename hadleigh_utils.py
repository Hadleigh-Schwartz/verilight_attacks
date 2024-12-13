"""
Mostly tweaked versions of utils from submodules
"""
from torchvision import transforms
import mediapipe as mp
import numpy as np
import cv2

import sys

sys.path.append("deconstruct-mediapipe/")
from test_converted_model import init_mpipe_blendshapes_model

def pad_image(im, desired_size=192):
    """
    Modified version of pad_image in mediapipe_pytorch/facial_landmarks/utils.py 
    to work with torch tensors

    The cv2.resize function is replaced with torchvision.transforms.Resize
        cv2.copyMakeBorder is replaced with transforms.Pad
    """
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    # im = cv2.resize(im, (new_size[1], new_size[0]))

    im = im.permute(2, 0, 1)
    im = transforms.Resize(new_size)(im)
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    # color = [0, 0, 0]
    # new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
    #     value=color)
    new_im = transforms.Pad([left, top, right, bottom])(im)
    new_im = new_im.permute(1, 2, 0)
    return new_im

def get_real_mediapipe_results(img_path):
    mesh_detector = init_mpipe_blendshapes_model(task_path = "deconstruct-mediapipe/face_landmarker_v2_with_blendshapes.task")
    image_mp = mp.Image.create_from_file(img_path)
    # mp_blob = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_face_array)
    mesh_results = mesh_detector.detect(image_mp)
    # Convert landmarks to numpy
    landmarks_np = []
    for face_idx in range(len(mesh_results.face_landmarks)):
        landmarks_np.append(
            np.array([[i.x, i.y, i.z] for i in mesh_results.face_landmarks[face_idx]])
        )
    landmarks_np = np.array(landmarks_np).astype("float32")

    landmarks_np = landmarks_np[0, :, :]
    image_cv = cv2.imread(img_path)
    for i in range(landmarks_np.shape[0]):
        coord = landmarks_np[i, :]
        x, y = coord[0], coord[1]
        cv2.circle(image_cv, (int(x*image_cv.shape[1]), int(y*image_cv.shape[0])), 1, (0, 255, 0), -1)
    cv2.imshow("gt landmarks", image_cv)
    cv2.waitKey(0)