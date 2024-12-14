""""
Notes on submodule modifications:
- Modified facenet-pytorch/models/mtcnn/detect to toggle off the torch.no_grad
- Commented out torch.no_grads() in mediapipe_pytorch/facial_landmarks/facial_lm_model.py/forward
- Commented out torch.no_grads() in mediapipe_pytorch/iris/irismodel.py/forward
"""


import sys
import cv2
import torch
from torch import nn
import numpy as np
from hadleigh_utils import pad_image, get_real_mediapipe_results
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("facenet-pytorch")
from models.mtcnn import MTCNN

sys.path.append("mediapipe_pytorch/facial_landmarks")
from facial_lm_model import FacialLM_Model 

sys.path.append("mediapipe_pytorch/iris")
from irismodel import IrisLM

sys.path.append("deconstruct-mediapipe/")
from blendshape_info import BLENDSHAPE_MODEL_LANDMARKS_SUBSET,  BLENDSHAPE_NAMES
from mlp_mixer import MediaPipeBlendshapesMLPMixer

class PyTorchMediapipeFaceMesh(nn.Module):
    def __init__(self):
        super(PyTorchMediapipeFaceMesh, self).__init__()
        self.mtcnn = MTCNN()
        self.facelandmarker = FacialLM_Model()
        self.facelandmarker_weights = torch.load('mediapipe_pytorch/facial_landmarks/model_weights/facial_landmarks.pth')
        self.facelandmarker.load_state_dict(self.facelandmarker_weights)
        self.facelandmarker = self.facelandmarker.eval() # is this strictly necessary or not allowed?

        self.irislandmarker = IrisLM()
        self.irislandmarker_weights = torch.load('mediapipe_pytorch/iris/model_weights/irislandmarks.pth')
        self.irislandmarker.load_state_dict(self.irislandmarker_weights)
        self.irislandmarker = self.irislandmarker.eval()  # is this strictly necessary or not allowed?

        self.blendshape_model = MediaPipeBlendshapesMLPMixer()
        self.blendshape_model.load_state_dict(torch.load("deconstruct-mediapipe/face_blendshapes.pth"))

    def detect_face(self, img, padding = 50):
        """
        Detects face in image using MTCNN and returns cropped face tensor

        Parameters:
            img: torch.Tensor
                The image to detect the face in
            padding: int
                The amount of padding to add around the detected face
        
        Returns:
            torch.Tensor
                Image cropped to detected face
        """
        bboxes, probs = self.mtcnn.detect(img)
        bbox = bboxes[0]
        bbox = bbox + [-padding, -padding, padding, padding] # add padding to the bbox, based on observation that mediapipe landmarks extractor benefits from this
        x1, y1, x2, y2 = bbox.astype(int)
        cropped_face_tensor = img_tensor[y1:y2, x1:x2, :]
        return cropped_face_tensor

    def preprocess_face_for_landmark_detection(self, img):
        """
        Apply preprocessing steps required by mediapipe_pytorch facial landmarker model. 
        See example usage in mediapipe_pytorch/facial_landmarks/inference.py to understand why this is required.

        Parameters:
            img: torch.Tensor
                Image of face to preprocess
        
        Returns:
            torch.Tensor
                Preprocessed face image
        """
        proc_img = pad_image(img, desired_size=192) # resize/pad
        proc_img = (proc_img / 127.5) - 1.0 # normalize 
        proc_img = proc_img.permute(2, 0, 1) # per line 116 of mediapipe_pytorch/facial_landmarks/inference.py, blob is expected to have dimensions [3, H, W]
        return proc_img

    def unnormalize_face(self, img):
        """
        Undo the normalization and permutation applied for mediapipe-pytorch facial landmark detection, but leave the padding it added. 
        This is for obtaining an image that can be displayed for visualization purposes.

        Parameters:
            img: torch.Tensor
                Image of face, previously prepreocessed using preprocess_face_for_landmark_detection
        
        Returns:
            torch.Tensor
                Unnormalized face image
        """
        img = img.permute(1, 2, 0)
        img = (img + 1.0) * 127.5
        return img

    def preprocess_eye_for_landmark_detection(self, img):
        """
        Apply preprocessing steps required by mediapipe_pytorch iris landmarker model.
        See example usage in mediapipe_pytorch/iris/inference.py to understand why this is required.

        Parameters:
            img: torch.Tensor
                Image of eye to preprocess
        """
        # preprocess img
        proc_img = pad_image(img, desired_size=64) # resize
        proc_img /= 255 # normalize
        proc_img = proc_img.permute(2, 0, 1) # per line 116 of mediapipe_pytorch/facial_landmarks/inference.py, blob is expected to have dimensions [3, H, W]
        return proc_img

    def unnormalize_eye(self, img):
        """
        Undo the normalization and permutation applied for mediapipe-pytorch iris landmark detection, but leave the padding it added.
        This is for obtaining an image that can be displayed for visualization purposes.

        Parameters:
            img: torch.Tensor
                Image of eye, previously preprocessed using preprocess_eye_for_landmark_detection
        Returns:
            torch.Tensor
                Unnormalized eye image
        """
        img *= 255
        img = img.permute(1, 2, 0)
        return img


    def get_eye_bounds(self, facial_landmarks, eye = "left"):
        """
        Compute boundaries of the eyes based on the facial landmarks.
        Based off of this resource (https://github.com/Morris88826/MediaPipe_Iris/blob/main/README.md), the iris model expects a 64x64 image of the eye
        cropped according to the  (rightEyeUpper0, rightEyeLower0, leftEyeUpper0, leftEyeLower0) landmarks, whose indices are specified 
        here: https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
        It is also mentioned that a significant (0.25 or 1, the README is ambiguous), is added to the crop of the eye. I will add 0.5*width to the left and right,
        and 0.5*height to the top and bottom as a middle ground.
        """
        
        rightEyeUpper0_ids = [246, 161, 160, 159, 158, 157, 173] # landmark indices for the upper eye contour of the right eye
        rightEyeLower0_ids = [33, 7, 163, 144, 145, 153, 154, 155, 133]

        leftEyeUpper0_ids = [466, 388, 387, 386, 385, 384, 398] # landmark indices for the upper eye contour of the left eye
        leftEyeLower0_ids = [263, 249, 390, 373, 374, 380, 381, 382, 362]

        rightEyeUpper0 = facial_landmarks[rightEyeUpper0_ids, :] # get the relevant eye landmark coordinates 
        rightEyeLower0 = facial_landmarks[rightEyeLower0_ids, :]
        leftEyeUpper0 = facial_landmarks[leftEyeUpper0_ids, :]
        leftEyeLower0 = facial_landmarks[leftEyeLower0_ids, :]

        if eye == "right":
            eye_left = rightEyeUpper0[:, 0].min().item()
            eye_right = rightEyeUpper0[:, 0].max().item()
            eye_top = rightEyeUpper0[:, 1].min().item()
            eye_bottom = rightEyeLower0[:, 1].max().item()
        else:
            eye_left = leftEyeUpper0[:, 0].min().item()
            eye_right = leftEyeUpper0[:, 0].max().item()
            eye_top = leftEyeUpper0[:, 1].min().item()
            eye_bottom = leftEyeLower0[:, 1].max().item()

        eye_width = eye_right - eye_left
        horizontal_margin = 0.5 * eye_width
        eye_left -= horizontal_margin
        eye_right += horizontal_margin

        eye_height = eye_bottom - eye_top
        vertical_margin = 0.5 * eye_height
        eye_top -= vertical_margin
        eye_bottom += vertical_margin

        return eye_left, eye_right, eye_top, eye_bottom, eye_width, eye_height

    def vis_facial_landmarks(self, img, facial_landmarks):
        """
        img: tensor, H x W x 3, RGB
        """
        img = img.numpy().astype(np.uint8)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for i in range(facial_landmarks.shape[0]):
            coord = facial_landmarks[i, :]
            x, y = coord[0].item(), coord[1].item()
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
        cv2.imshow("Facial Landmarks", img)
        cv2.waitKey(0)

    def vis_eye_cropping(self, img, right_eye_bounds, left_eye_bounds):
        """
        img: tensor, H x W x 3, RGB
        """
        img = img.numpy().astype(np.uint8)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        right_eye_left, right_eye_right, right_eye_top, right_eye_bottom = right_eye_bounds
        left_eye_left, left_eye_right, left_eye_top, left_eye_bottom = left_eye_bounds

        cv2.rectangle(img, (int(right_eye_left), int(right_eye_top)), (int(right_eye_right), int(right_eye_bottom)), (0, 255, 0), 2)
        cv2.rectangle(img, (int(left_eye_left), int(left_eye_top)), (int(left_eye_right), int(left_eye_bottom)), (0, 255, 0), 2)

        cv2.imshow("Eye Cropping", img)
        cv2.waitKey(0)

    def vis_iris_landmarks_on_eye_crop(self, img, iris_landmarks):
        """
        img: eye crop tensor, 64 x 64 x 3, RGB
        """
        img = img.numpy().astype(np.uint8)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for i in range(iris_landmarks.shape[0]):
            coord = iris_landmarks[i, :]
            x, y = coord[0].item(), coord[1].item()
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
        cv2.imshow("Iris Landmarks", img)
        cv2.waitKey(0)

    def vis_iris_landmarks_on_face(self, img, iris_landmarks):
        """
        img: face tensor, H x W x 3, RGB
        """
        img = img.numpy().astype(np.uint8)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for i in range(iris_landmarks.shape[0]):
            coord = iris_landmarks[i, :]
            x, y = coord[0].item(), coord[1].item()
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
        cv2.imshow("Iris Landmarks", img)
        cv2.waitKey(0)

    def iris_landmarks_to_original_pixel_space(self, iris_landmarks, eye_bounds, eye_crop_dims):
        """
        The iris landmarks were computed on a 64x64 padded image of the eye crop, obtained by extracting a crop from the 192x192
        and then resizing in pad_image.
        
        We need to convert these eye-crop-centeric landmarks to be in same pixel space that the facial landmarks were computed in, 
        so we first unscale the landmarks back to the 192x192 space, then adjust them according to the boundaries of the eye crop
        w.r.t to the 192x192 image. 

        iris_landmarks: tensor, 5 x 3
        eye_bounds: tuple, (left, right, top, bottom)
        """
        left, right, top, bottom = eye_bounds
        eye_width, eye_height = eye_crop_dims
        ratio = float(64)/max(eye_height*2, eye_width*2)
        new_eye_size = tuple([int(x*ratio) for x in [eye_height*2, eye_width*2]])
        eye_delta_w = 64 - new_eye_size[1]
        eye_delta_h = 64 - new_eye_size[0]
        eye_top_padding, eye_bottom_padding = eye_delta_h//2, eye_delta_h-(eye_delta_h//2)
        eye_left_padding, eye_right_padding = eye_delta_w//2, eye_delta_w-(eye_delta_w//2)

        iris_landmarks[:, 0] = iris_landmarks[:, 0] - eye_left_padding
        iris_landmarks[:, 0] = iris_landmarks[:, 0]*(eye_width*2/new_eye_size[1]) + left
        iris_landmarks[:, 1] = iris_landmarks[:, 1] - eye_top_padding
        iris_landmarks[:, 1] = iris_landmarks[:, 1]*(eye_height*2/new_eye_size[0]) + top

        return iris_landmarks

    def compare_to_real_mediapipe(self, landmarks, blendshapes, padded_face):
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
        real_mp_landmarks, real_mp_blendshapes = get_real_mediapipe_results(padded_face)
        real_mp_blendshapes = real_mp_blendshapes.round(3)
        blendshapes = blendshapes.round(3)

        # now convert padded_face to BGR for below opencv stuff
        padded_face = cv2.cvtColor(padded_face, cv2.COLOR_RGB2BGR)

        # make blown up visualization of landmarks for comparison of all 478
        scaled_real_mp_landmarks = real_mp_landmarks[:, :2] * 35
        scaled_landmarks = landmarks[:, :2]* 35
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
        for i in range(landmarks.shape[0]):
            coord = landmarks[i, :]
            x, y = coord[0], coord[1]
            cv2.circle(annotated_face_ours, (int(x), int(y)), 1, (0, 0, 255), -1)
        

        # concatenate the blendshapes to the image
        fig_arr_shape = fig_arr.shape
        annotated_face_shape = annotated_face_real.shape

        if annotated_face_shape[0]*2 < fig_arr_shape[0] and annotated_face_shape[1]*2 < fig_arr_shape[1]:
            scaling_ratio = min(fig_arr_shape[0]/annotated_face_shape[0], fig_arr_shape[1]/annotated_face_shape[1])
            annotated_face_real = cv2.resize(annotated_face_real, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
            annotated_face_ours = cv2.resize(annotated_face_ours, (0, 0), fx=scaling_ratio, fy=scaling_ratio)
        
        # update annotated_face_shape
        annotated_face_shape = annotated_face_real.shape

        if fig_arr_shape[0] > annotated_face_shape[0]:
            # add padding to the image
            padding = fig_arr_shape[0] - annotated_face_shape[0]
            annotated_face_real = np.pad(annotated_face_real, ((0, padding), (0, 0), (0, 0)), mode='constant')
            annotated_face_ours = np.pad(annotated_face_ours, ((0, padding), (0, 0), (0, 0)), mode='constant')
        elif fig_arr_shape[0] < annotated_face_shape[0]:
            padding = annotated_face_shape[0] - fig_arr_shape[0]
            fig_arr = np.pad(fig_arr, ((0, padding), (0, 0), (0, 0)), mode='constant')

        img = np.hstack((annotated_face_ours, annotated_face_real, fig_arr))
        cv2.imshow("Final Comp", img)
        cv2.waitKey(0)

    def forward(self, img_tensor):
        # Note: detect_face uses with torch.no_grad() internally. Is that a problem?
        face = self.detect_face(img_tensor)
        print("Face detect grad: " , face.grad_fn)
        proc_face = self.preprocess_face_for_landmark_detection(face)
        print("Proc face grad: " , proc_face.grad_fn)

        # run facial landmark detection
        facial_landmarks, confidence = self.facelandmarker.predict(proc_face) # predict
        print("Facial landmarks grad: " , facial_landmarks.grad_fn)
        facial_landmarks = facial_landmarks[0, :, :, :] # assume there is only one face in the image, so take the first set of landmarks
        facial_landmarks = facial_landmarks.view(468, 3)
        print("Facial landmarks grad: " , facial_landmarks.grad_fn)
        
        # only for visualization
        padded_face = self.unnormalize_face(proc_face) # undo normalization and permutation applied for facial landmark detection, but leave the padding it added
        # vis_facial_landmarks(padded_face, facial_landmarks) # debugging only

        # get eye bounds
        right_eye_left, right_eye_right, right_eye_top, right_eye_bottom, right_eye_width, right_eye_height = self.get_eye_bounds(facial_landmarks, eye = "right")
        left_eye_left, left_eye_right, left_eye_top, left_eye_bottom, left_eye_width, left_eye_height = self.get_eye_bounds(facial_landmarks, eye = "left")
   
        left_eye_crop = padded_face[int(left_eye_top):int(left_eye_bottom), int(left_eye_left):int(left_eye_right), :]
        right_eye_crop = padded_face[int(right_eye_top):int(right_eye_bottom), int(right_eye_left):int(right_eye_right), :]
        print("Left eye crop grad: " , left_eye_crop.grad_fn, "Right eye crop grad: " , right_eye_crop.grad_fn)

        # debugging only
        # vis_eye_cropping(padded_face, facial_landmarks, (right_eye_left, right_eye_right, right_eye_top, right_eye_bottom), (left_eye_left, left_eye_right, left_eye_top, left_eye_bottom))

        # pad the eye crops
        proc_left_eye_crop = self.preprocess_eye_for_landmark_detection(left_eye_crop)
        proc_right_eye_crop = self.preprocess_eye_for_landmark_detection(right_eye_crop)
        print("Proc left eye crop grad: " , proc_left_eye_crop.grad_fn, "Proc right eye crop grad: " , proc_right_eye_crop.grad_fn)

        # run iris detection
        left_eye_contour_landmarks, left_iris_landmarks = self.irislandmarker.predict(proc_left_eye_crop)
        right_eye_contour_landmarks, right_iris_landmarks = self.irislandmarker.predict(proc_right_eye_crop)
        left_iris_landmarks = left_iris_landmarks.view(5, 3)
        right_iris_landmarks = right_iris_landmarks.view(5, 3)
        print("Left iris landmarks grad: " , left_iris_landmarks.grad_fn, "Right iris landmarks grad: " , right_iris_landmarks.grad_fn)

        # debugging only
        # padded_left_eye = unnormalize_eye(proc_left_eye_crop)
        # padded_right_eye = unnormalize_eye(proc_right_eye_crop)
        # vis_iris_landmarks_on_eye_crop(padded_left_eye, left_iris_landmarks) 
        # vis_iris_landmarks_on_eye_crop(padded_right_eye, right_iris_landmarks)

        # adjust the iris landmarks to the original image pixel space
        left_iris_landmarks = self.iris_landmarks_to_original_pixel_space(left_iris_landmarks, (left_eye_left, left_eye_right, left_eye_top, left_eye_bottom), (left_eye_width, left_eye_height))
        right_iris_landmarks = self.iris_landmarks_to_original_pixel_space(right_iris_landmarks, (right_eye_left, right_eye_right, right_eye_top, right_eye_bottom), (right_eye_width, right_eye_height))
        print("Left iris landmarks original space grad: " , left_iris_landmarks.grad_fn, "Right iris landmarks original space grad: " , right_iris_landmarks.grad_fn)

        # only for visualization
        # self.vis_iris_landmarks_on_face(padded_face, left_iris_landmarks)
        # self.vis_iris_landmarks_on_face(padded_face, right_iris_landmarks)

        # append the iris landmarks to the general landmarks array in the proper order
        # know from https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
        # [468, 469, 470, 471, 472] are left iris landmarks
        # confirmed by analyzing and comparing to here https://github.com/k-m-irfan/simplified_mediapipe_face_landmarks
        # that the indices of the left iris landmarks are 468, 469, 470, 471, 472, right iris landmarks are 473, 474, 475, 476, 477
        # (i.e., ordered same as in the MediaPipe 478-landmarker model), therefore can just append in order, left first, then right
        # IMPORTANT: SWAP LEFT AND RIGHT IRIS LANDMARKS. Left crop actually corresponds to what MP considers right eye, because of mirrorring.This is confirmed via the zoomed-in visualization
        # comparing actual MediaPipe 478-landmarker model output to the output of our piecemeal facial then iris landmark detection approach
        all_landmarks = torch.cat([facial_landmarks, right_iris_landmarks, left_iris_landmarks], dim=0)

        # # unlike in case of landmarks outputted by the actual mediapipe model, ours currently are already
        # # scaled by the image dimensions, so no need to perform line 96 in deconstruct-mediapipe/test_converted_model.py
        blendshape_input = all_landmarks[BLENDSHAPE_MODEL_LANDMARKS_SUBSET, :2]
        blendshape_input = blendshape_input.unsqueeze(0) # add batch dimension
        # with torch.no_grad(): # is this necessary or strictly not allowed?
        blendshapes = self.blendshape_model(blendshape_input)
        blendshapes = blendshapes.squeeze()
        print("Blendshapes grad: " , blendshapes.grad_fn)
        return all_landmarks, blendshapes, padded_face

mp = PyTorchMediapipeFaceMesh()

img_path = "obama2.jpeg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.tensor(img, dtype=torch.float32, requires_grad = True) # emulate format that will be output by generator
landmarks, blendshapes, padded_face = mp(img_tensor)
print(landmarks.grad_fn, blendshapes.grad_fn)

padded_face = padded_face.detach().numpy().astype(np.uint8)
landmarks_np = landmarks.detach().numpy()
blendshapes_np = blendshapes.detach().numpy()
mp.compare_to_real_mediapipe(landmarks_np, blendshapes_np, padded_face)


# cap = cv2.VideoCapture(0)
# count = 0
# img_path = "curr.jpg"
# ret, img = cap.read()
# if not ret:
#     break
# count += 1
# if count < 4:
#     continue
# save as curr.jpg
# cv2.imwrite(img_path, img)

