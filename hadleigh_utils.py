
from torchvision import transforms
import mediapipe as mp
import numpy as np
import cv2
import tensorflow as tf
from mp_alignment_differentiable import align_landmarks as align_landmarks_differentiable
from mp_alignment_original import align_landmarks as align_landmarks_original
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d    

import sys

sys.path.append("facenet-pytorch")
from models.mtcnn import MTCNN

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


def record_face_video():
    """
    Record and save a video on the webcam. Testing purposes
    """
    mtcnn = MTCNN(thresholds=[0.4, 0.4, 0.4]) 
        

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    out_frames = []
    num_frames = 90
    count = 0
    warmup_count = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        warmup_count += 1
        if warmup_count < 10:
            continue
        count += 1
        out_frames.append(img)
        if count == num_frames:
            break
    cap.release()
    print("Finished recording. Now processing...")

    all_bboxes = []
    face_frames = []
    padding = 50
    for frame in out_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes, probs = mtcnn.detect(frame)
        if bboxes is None:
            print("No face detected")
            continue
        face_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        face_frames.append(face_frame)
        bbox = bboxes[0]
        bbox = bbox + [-padding, -padding, padding, padding] # add padding to the bbox, based on observation that mediapipe landmarks extractor benefits from this
        all_bboxes.append(bbox)
        # x1, y1, x2, y2 = bbox.astype(int)
        # cropped_face_tensor = img[y1:y2, x1:x2, :]
    all_bboxes = np.array(all_bboxes)
    widths = all_bboxes[:, 2] - all_bboxes[:, 0]
    heights = all_bboxes[:, 3] - all_bboxes[:, 1]
    max_width = int(np.max(widths))
    max_height = int(np.max(heights))

    print("Writing video...")
    out = cv2.VideoWriter('test_video.mp4', fourcc, 20, (max_width, max_height))
    for i, frame in enumerate(face_frames):
        bbox = all_bboxes[i]
        x1, y1, x2, y2 = bbox.astype(int)
        cropped_face = frame[y1:y2, x1:x2, :]
        width = x2 - x1
        height = y2 - y1
        padding_x = int((max_width - width) / 2)
        padding_y = int((max_height - height) / 2)
        x_diff = 0
        y_diff = 0
        if padding_x*2 + width != max_width:
            x_diff = max_width - (padding_x*2 + width)
        if padding_y*2 + height != max_height:
            y_diff = max_height - (padding_y*2 + height)
        # add the diff to left
        padded_face = np.pad(cropped_face, ((padding_y + y_diff, padding_y), (padding_x + x_diff, padding_x), (0, 0)))
        # print(f"{i}/{len(face_frames)}, {padded_face.shape}, {padded_face.dtype}")
        out.write(padded_face)

    out.release()

def compare_to_real_mediapipe(landmarks_torch,  blendshapes, padded_face, save_landmark_comparison = False, live_demo = False):
    """

    Parameters:
        landmarks: np.ndarray, 468 x 3
            The facial landmarks outputted by our model
        blendshapes: np.ndarray, 52, 
            The blendshapes outputted by our model
        padded_face : np.ndarray, HxWx3 in RGB format
            Image of the face that was padded and preprocessed for the facial landmark detection. 
            All landmarks are relative to this image, so we must use it to visualize the landmarks and blendshapes outputted by our model.
    
    Returns:
        None
    """
    landmarks_np_ours = landmarks_torch.clone().detach().numpy()
    real_mp_landmarks, real_mp_blendshapes = get_real_mediapipe_results(padded_face)
    if real_mp_landmarks is None and real_mp_blendshapes is None:
        print("NO FACE DETECTED BY REAL MEDIAPIPE")
        return

    W, H = padded_face.shape[1], padded_face.shape[0]
    aligned3d_real, aligned2d_real = align_landmarks_original(real_mp_landmarks, W, H, W, H)
    aligned3d_real = np.array(aligned3d_real)
    aligned2d_real = np.array(aligned2d_real)
    aligned3d_ours, aligned2d_ours = align_landmarks_differentiable(landmarks_torch, W, H, W, H)

    real_mp_blendshapes = real_mp_blendshapes.round(3)
    blendshapes = blendshapes.round(3)

    # # visualize point clouds of aligned landmarks, real and ours
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(aligned_real)
    # o3d.visualization.draw_geometries([pc])

    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(aligned_ours.detach().numpy())
    # o3d.visualization.draw_geometries([pc])

    # now convert padded_face to BGR for below opencv stuff
    padded_face = cv2.cvtColor(padded_face, cv2.COLOR_RGB2BGR)

    if save_landmark_comparison:
        # make blown up visualization of landmarks for comparison of all 478
        scaled_real_mp_landmarks = real_mp_landmarks[:, :2] * 35
        scaled_landmarks = landmarks_np_ours[:, :2]* 35
        blank = np.ones((6000, 6000, 3), dtype=np.uint8) * 255
        for i in range(scaled_real_mp_landmarks.shape[0]):
            coord = scaled_real_mp_landmarks[i, :]
            x, y = coord[0], coord[1]
            cv2.circle(blank, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(blank, "Real" + str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for i in range(scaled_landmarks.shape[0]):
            coord = scaled_landmarks[i, :]
            x, y = coord[0], coord[1]
            cv2.circle(blank, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.putText(blank, "Our Model" + str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite("landmarks_comparison.png", blank)

    # create bar plots for the blendshapes
    plt.style.use('ggplot')
    plt.figsize=(20, 20)
    df = pd.DataFrame(data={'Our Model': blendshapes, 'Real MP': real_mp_blendshapes}, 
                        index=BLENDSHAPE_NAMES)
    df.plot(kind='bar', color = ['r', 'g'])
    plt.tight_layout()
    plt.draw()
    fig = plt.gcf()
    fig.canvas.draw()
    fig_arr = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    fig_arr = fig_arr[:, :, ::-1]
    plt.close()


    # make annotated faces
    annotated_face_real = padded_face.copy()
    for i in range(real_mp_landmarks.shape[0]):
        coord = real_mp_landmarks[i, :]
        x, y = coord[0], coord[1]
        cv2.circle(annotated_face_real, (int(x), int(y)), 1, (0, 255, 0), -1)
    
    annotated_face_ours = padded_face.copy()
    for i in range(landmarks_np_ours.shape[0]):
        coord = landmarks_np_ours[i, :]
        x, y = coord[0], coord[1]
        cv2.circle(annotated_face_ours, (int(x), int(y)), 1, (0, 0, 255), -1)
    
    # make annotated blank images of aligned landmarks
    annotated_aligned_landmarks2d_real = np.ones((H, W, 3), dtype=np.uint8) * 255
    for i in range(aligned2d_real.shape[0]):
        coord = aligned2d_real[i, :]
        x, y = coord[0], coord[1]
        cv2.circle(annotated_aligned_landmarks2d_real, (int(x), int(y)), 1, (0, 255, 0), -1)
    annotated_aligned_landmarks2d_ours = np.ones((H, W, 3), dtype=np.uint8) * 255
    for i in range(aligned2d_ours.shape[0]):
        coord = aligned2d_ours[i, :]
        x, y = coord[0], coord[1]
        cv2.circle(annotated_aligned_landmarks2d_ours, (int(x), int(y)), 1, (0, 0, 255), -1)

    # concatenate the annotated faces to the annotated aligned landmarks
    annotated_real = np.vstack((annotated_face_real, annotated_aligned_landmarks2d_real))
    annotated_ours = np.vstack((annotated_face_ours, annotated_aligned_landmarks2d_ours))
    

    # concatenate the blendshapes to the image
    fig_arr_shape = fig_arr.shape
    annotated_real_shape = annotated_real.shape

    if annotated_real_shape[0]*2 < fig_arr_shape[0] and annotated_real_shape[1]*2 < fig_arr_shape[1]:
        scaling_ratio = min(fig_arr_shape[0]/annotated_real_shape[0], fig_arr_shape[1]/annotated_real_shape[1])
        annotated_real = cv2.resize(annotated_real, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
        annotated_ours = cv2.resize(annotated_ours, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
    
    # update annotated_real_shape
    annotated_real_shape = annotated_real.shape

    if fig_arr_shape[0] > annotated_real_shape[0]:
        # add padding to the image
        padding = fig_arr_shape[0] - annotated_real_shape[0]
        annotated_real = np.pad(annotated_real, ((0, padding), (0, 0), (0, 0)), mode='constant')
        annotated_ours = np.pad(annotated_ours, ((0, padding), (0, 0), (0, 0)), mode='constant')
    elif fig_arr_shape[0] < annotated_real_shape[0]:
        padding = annotated_real_shape[0] - fig_arr_shape[0]
        fig_arr = np.pad(fig_arr, ((0, padding), (0, 0), (0, 0)), mode='constant')

    img = np.hstack((annotated_ours, annotated_real, fig_arr))
    cv2.imshow("Final Comp", img)
    if live_demo:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
    else:
        cv2.waitKey(0)
