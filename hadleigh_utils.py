"""
Mostly tweaked versions of utils from submodules
"""
from torchvision import transforms
import mediapipe as mp
import numpy as np
import cv2
import tensorflow as tf

import sys

sys.path.append("deconstruct-mediapipe/")
from test_converted_model import init_mpipe_blendshapes_model, get_blendshape_score_by_index
from blendshape_info import BLENDSHAPE_NAMES

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

def get_real_mediapipe_results(img):
    """
    Compute the landmarks and blendshapes from the image using the actual Google mediapipe model

    Parameters:
    img: np.ndarray
        The image to process
    
    Returns:
    landmarks_np: np.ndarray
        The landmarks detected for the first face
    blendshapes_np: np.ndarray
        The blendshape scores detected for the first face
    """
    mesh_detector = init_mpipe_blendshapes_model(task_path = "deconstruct-mediapipe/face_landmarker_v2_with_blendshapes.task")
    img = np.ascontiguousarray(img, dtype=np.uint8)
    image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    mesh_results = mesh_detector.detect(image_mp)

    if len(mesh_results.face_landmarks) == 0:
        return None, None
    
    # Convert landmarks to numpy
    landmarks_np = []
    for face_idx in range(len(mesh_results.face_landmarks)):
        landmarks_np.append(
            np.array([[i.x*image_mp.width, i.y*image_mp.height, i.z] for i in mesh_results.face_landmarks[face_idx]])
        )
    landmarks_np = np.array(landmarks_np).astype("float32")

    landmarks_np = landmarks_np[0, :, :]

    blendshapes_np = np.array(
        [
            [
                get_blendshape_score_by_index(
                    mesh_results.face_blendshapes[face_idx], i
                )
                for i in range(len(BLENDSHAPE_NAMES))
            ]
            for face_idx in range(len(mesh_results.face_landmarks))
        ]
    )
    blendshapes_np = blendshapes_np[0, :]
    return landmarks_np, blendshapes_np
    

class TFLiteModel:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, *data_args):
        assert len(data_args) == len(self.input_details)
        for data, details in zip(data_args, self.input_details):
            self.interpreter.set_tensor(details["index"], data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])
