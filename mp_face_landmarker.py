""""
Notes on submodule modifications:
- Commented out torch.no_grads() in mediapipe_pytorch/facial_landmarks/facial_lm_model.py/forward
- Commented out torch.no_grads() in mediapipe_pytorch/iris/irismodel.py/forward
"""


import sys
import cv2
import torch
from torch import nn
import numpy as np
from hadleigh_utils import pad_image, get_real_mediapipe_results, compare_to_real_mediapipe, record_face_video
from mp_alignment_differentiable import align_landmarks as align_landmarks_differentiable
from mp_alignment_original import align_landmarks as align_landmarks_original
from torchviz import make_dot
from matplotlib import pyplot as plt

sys.path.append("facenet-pytorch")
from models.mtcnn import MTCNN

sys.path.append("mediapipe_pytorch/facial_landmarks")
from facial_lm_model import FacialLM_Model 

sys.path.append("mediapipe_pytorch/iris")
from irismodel import IrisLM

sys.path.append("deconstruct-mediapipe/")
from blendshape_info import BLENDSHAPE_MODEL_LANDMARKS_SUBSET,  BLENDSHAPE_NAMES
from mlp_mixer import MediaPipeBlendshapesMLPMixer

class PyTorchMediapipeFaceLandmarker(nn.Module):
    def __init__(self):
        super(PyTorchMediapipeFaceLandmarker, self).__init__()
        
        # face detection
        self.mtcnn = MTCNN(thresholds = [0.4, 0.4, 0.4]) # more generous thresholds to ensure face is detected
        
        # facial landmarks
        self.facelandmarker = FacialLM_Model()
        self.facelandmarker_weights = torch.load('mediapipe_pytorch/facial_landmarks/model_weights/facial_landmarks.pth')
        self.facelandmarker.load_state_dict(self.facelandmarker_weights)
        self.facelandmarker = self.facelandmarker.eval() 

        # iris landmarks
        self.irislandmarker = IrisLM()
        self.irislandmarker_weights = torch.load('mediapipe_pytorch/iris/model_weights/irislandmarks.pth')
        self.irislandmarker.load_state_dict(self.irislandmarker_weights)
        self.irislandmarker = self.irislandmarker.eval() 

        # blendshapes from facial and iris landmarks
        self.blendshape_model = MediaPipeBlendshapesMLPMixer()
        self.blendshape_model.load_state_dict(torch.load("deconstruct-mediapipe/face_blendshapes.pth"))

    def detect_face(self, img, padding = 50):
        """
        Detects face in image using MTCNN and returns cropped face tensor

        WARNING: This is not differentiable, it's just here for
        the purpose of checking if a face was detected.

        Parameters:
            img: torch.Tensor H x W x 3
                The image to detect the face in
            padding: int
                The amount of padding to add around the detected face
        
        Returns:
            torch.Tensor
                Image cropped to detected face
        """
        bboxes, probs = self.mtcnn.detect(img)
        if bboxes is None: # if no face detected
            return torch.ones(img.shape, dtype=torch.float32)*-1 # placeholder for Nans
        bbox = bboxes[0]
        bbox = bbox + [-padding, -padding, padding, padding] # add padding to the bbox, based on observation that mediapipe landmarks extractor benefits from this
        x1, y1, x2, y2 = bbox.astype(int)
        cropped_face_tensor = img[y1:y2, x1:x2, :]
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
        It is also mentioned that a significant (0.25 or 1, the README is ambiguous), is added to the crop of the eye.
        I noticed that it is essential to have the full eye lids and eyebrows included for the pruposes of blinking detection, 
        so unlike the resource above I also consider the eyebrow landmarks when determining the boundaries for the eye crop.
        I will add 0.25*width to the left and right, and .25*height to the top and bottom of the eye crop."""
        
        # this resources https://github.com/google-ai-edge/mediapipe/blob/ab1de4fced96c18b79eda4ab407b04eb08301ea4/mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt
        # considers 33 and 133 as denoting the boundaries of the left eye, and 263 and 362 as the right
        # i think it then sets the height equal to the width as computed by only these two guys. Let's try this.
        right_eye_ids = [362, 263]
        left_eye_ids = [33, 133]
        if eye == "right":
            eye_left = facial_landmarks[right_eye_ids[0], 0].item()
            eye_right = facial_landmarks[right_eye_ids[1], 0].item()
            eye_top = facial_landmarks[right_eye_ids, 1].min().item()
            eye_bottom = facial_landmarks[right_eye_ids, 1].max().item()    
        if eye == "left":
            eye_left = facial_landmarks[left_eye_ids[0], 0].item()
            eye_right = facial_landmarks[left_eye_ids[1], 0].item()
            eye_top = facial_landmarks[left_eye_ids, 1].min().item()
            eye_bottom = facial_landmarks[left_eye_ids, 1].max().item()
        eye_width = eye_right - eye_left
        horizontal_margin = 0.25 * eye_width 
        eye_left -= horizontal_margin
        eye_right += horizontal_margin
        eye_top -= eye_width // 2
        eye_top -= horizontal_margin
        eye_bottom += eye_width // 2
        eye_bottom += horizontal_margin
        eye_width = eye_width + 2*horizontal_margin
        eye_height = eye_width
            

        # og mediapipe stuff
        # rightEyeUpper0_ids = [246, 161, 160, 159, 158, 157, 173] # landmark indices for the upper eye contour of the right eye
        # rightEyeLower0_ids = [33, 7, 163, 144, 145, 153, 154, 155, 133]
        # rightEyebrowUpper_ids =  [156, 70, 63, 105, 66, 107, 55, 193]
        # rightEyebrowLower_ids =  [35, 124, 46, 53, 52, 65]

        # leftEyeUpper0_ids = [466, 388, 387, 386, 385, 384, 398] # landmark indices for the upper eye contour of the left eye
        # leftEyeLower0_ids = [263, 249, 390, 373, 374, 380, 381, 382, 362]
        # leftEyebrowUpper_ids =  [383, 300, 293, 334, 296, 336, 285, 417]
        # leftEyebrowLower_ids = [265, 353, 276, 283, 282, 295]

        # rightEyeUpper0 = facial_landmarks[rightEyeUpper0_ids, :] # get the relevant eye landmark coordinates 
        # rightEyeLower0 = facial_landmarks[rightEyeLower0_ids, :]
        # rightEyebrowLower = facial_landmarks[rightEyebrowLower_ids, :]
        # rightEyebrowUpper = facial_landmarks[rightEyebrowUpper_ids, :]
        # right_eye_all = torch.cat([rightEyeUpper0, rightEyeLower0], dim=0)

        # leftEyeUpper0 = facial_landmarks[leftEyeUpper0_ids, :]
        # leftEyeLower0 = facial_landmarks[leftEyeLower0_ids, :]
        # leftEyebrowLower = facial_landmarks[leftEyebrowLower_ids, :]
        # leftEyebrowUpper = facial_landmarks[leftEyebrowUpper_ids, :]
        # left_eye_all = torch.cat([leftEyeUpper0, leftEyeLower0], dim=0)
        # if eye == "right":
        #     # eye_left = rightEyeUpper0[:, 0].min().item()
        #     # eye_right = rightEyeUpper0[:, 0].max().item()
        #     # eye_top = rightEyeUpper0[:, 1].min().item()
        #     # eye_bottom = rightEyeLower0[:, 1].max().item()
        #     eye_left = right_eye_all[:, 0].min().item()
        #     eye_right = right_eye_all[:, 0].max().item()
        #     eye_top = right_eye_all[:, 1].min().item()
        #     eye_bottom = right_eye_all[:, 1].max().item()
        # else:
        #     # eye_left = leftEyeUpper0[:, 0].min().item()
        #     # eye_right = leftEyeUpper0[:, 0].max().item()
        #     # eye_top = leftEyeUpper0[:, 1].min().item()
        #     # eye_bottom = leftEyeLower0[:, 1].max().item()
        #     eye_left = left_eye_all[:, 0].min().item()
        #     eye_right = left_eye_all[:, 0].max().item()
        #     eye_top = left_eye_all[:, 1].min().item()
        #     eye_bottom = left_eye_all[:, 1].max().item()

        # og 
        # eye_width = eye_right - eye_left
        # horizontal_margin = 0.25 * eye_width
        # eye_left -= horizontal_margin
        # eye_right += horizontal_margin

        # eye_height = eye_bottom - eye_top
        # vertical_margin = .25 * eye_height 
        # eye_top -= vertical_margin
        # eye_bottom += vertical_margin

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
        img = img.detach().numpy().astype(np.uint8)
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
        img = img.detach().numpy().astype(np.uint8)
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
        ratio = float(64)/max(eye_height, eye_width) #why *2
        new_eye_size = tuple([int(x*ratio) for x in [eye_height, eye_width]])
        eye_delta_w = 64 - new_eye_size[1]
        eye_delta_h = 64 - new_eye_size[0]
        eye_top_padding, eye_bottom_padding = eye_delta_h//2, eye_delta_h-(eye_delta_h//2)
        eye_left_padding, eye_right_padding = eye_delta_w//2, eye_delta_w-(eye_delta_w//2)

        iris_landmarks[:, 0] = iris_landmarks[:, 0] - eye_left_padding
        iris_landmarks[:, 0] = iris_landmarks[:, 0]*(eye_width/new_eye_size[1]) + left
        iris_landmarks[:, 1] = iris_landmarks[:, 1] - eye_top_padding
        iris_landmarks[:, 1] = iris_landmarks[:, 1]*(eye_height/new_eye_size[0]) + top
        return iris_landmarks


    def forward(self, img_tensor):
        """
        Parameters:
        img_tensor: torch.Tensor
            Image tensor, H x W x 3, RGB. Should be a crop of a face.

        Returns:
        landmarks: torch.Tensor
            478 x 3 tensor of FaceMesh dense landmarks. 
        blendshapes: torch.Tensor
            52 tensor of blendshape scores
        """

        face_check = self.detect_face(img_tensor)
        # check if face is all zeros, which indicates no face was detected.
        # this is not differentiable and the bbox isn't used. It's only here as a check to prevent 
        # running the facial landmark detection model on an image with no face, 
        # which would cause bizarre landmarks to be extracted with no recourse.
        if torch.all(face_check == 0):
            print("NO FACE DETECTED")
            landmarks_zeroes = torch.zeros(478, 3)
            blendshapes_zeroes = torch.zeros(52)
            face_zeroes = torch.zeros(img_tensor.shape)
            return landmarks_zeroes, blendshapes_zeroes, face_zeroes

        proc_face = self.preprocess_face_for_landmark_detection(img_tensor)

        # run facial landmark detection
        facial_landmarks, confidence = self.facelandmarker.predict(proc_face) # predict
        facial_landmarks = facial_landmarks[0, :, :, :] # assume there is only one face in the image, so take the first set of landmarks
        facial_landmarks = facial_landmarks.view(468, 3)
        
        # only for visualization
        padded_face = self.unnormalize_face(proc_face) # undo normalization and permutation applied for facial landmark detection, but leave the padding it added
        # vis_facial_landmarks(padded_face, facial_landmarks) # debugging only


        # get eye bounds
        right_eye_left, right_eye_right, right_eye_top, right_eye_bottom, right_eye_width, right_eye_height = self.get_eye_bounds(facial_landmarks, eye = "right")
        left_eye_left, left_eye_right, left_eye_top, left_eye_bottom, left_eye_width, left_eye_height = self.get_eye_bounds(facial_landmarks, eye = "left")

       
        left_eye_crop = padded_face[int(left_eye_top):int(left_eye_bottom), int(left_eye_left):int(left_eye_right), :]
        right_eye_crop = padded_face[int(right_eye_top):int(right_eye_bottom), int(right_eye_left):int(right_eye_right), :]
        # need to flip horizontally for right eye, according to https://github.com/Morris88826/MediaPipe_Iris
        right_eye_crop = torch.flip(right_eye_crop, [1])


        # debugging only
        # vis_eye_cropping(padded_face, facial_landmarks, (right_eye_left, right_eye_right, right_eye_top, right_eye_bottom), (left_eye_left, left_eye_right, left_eye_top, left_eye_bottom))
 
        # pad the eye crops
        proc_left_eye_crop = self.preprocess_eye_for_landmark_detection(left_eye_crop)
        proc_right_eye_crop = self.preprocess_eye_for_landmark_detection(right_eye_crop)

        # run iris detection
        left_eye_contour_landmarks, left_iris_landmarks = self.irislandmarker.predict(proc_left_eye_crop)
        right_eye_contour_landmarks, right_iris_landmarks = self.irislandmarker.predict(proc_right_eye_crop)
        left_iris_landmarks = left_iris_landmarks.view(5, 3)
        right_iris_landmarks = right_iris_landmarks.view(5, 3)
        left_eye_contour_landmarks = left_eye_contour_landmarks.view(71, 3)
        right_eye_contour_landmarks = right_eye_contour_landmarks.view(71, 3)

        # test replacing 

        # debugging only
        all_left_eye_landmarks = torch.cat([left_eye_contour_landmarks, left_iris_landmarks], dim=0)
        all_right_eye_landmarks = torch.cat([right_eye_contour_landmarks, right_iris_landmarks], dim=0)
        padded_left_eye = self.unnormalize_eye(proc_left_eye_crop)
        padded_right_eye = self.unnormalize_eye(proc_right_eye_crop)
        self.vis_iris_landmarks_on_eye_crop(padded_left_eye, all_left_eye_landmarks) 
        self.vis_iris_landmarks_on_eye_crop(padded_right_eye, all_right_eye_landmarks)

        # adjust the iris landmarks to the original image pixel space
        left_iris_landmarks = self.iris_landmarks_to_original_pixel_space(left_iris_landmarks, (left_eye_left, left_eye_right, left_eye_top, left_eye_bottom), (left_eye_width, left_eye_height))
        right_iris_landmarks = self.iris_landmarks_to_original_pixel_space(right_iris_landmarks, (right_eye_left, right_eye_right, right_eye_top, right_eye_bottom), (right_eye_width, right_eye_height))
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
        all_landmarks = torch.cat([facial_landmarks, left_iris_landmarks, right_iris_landmarks], dim=0)
        
        # The z coordinate of the landmarks is scaled by the height of the image, unlike in the case of the actual MediaPipe model.
        # To convert to the same scale as the actual MediaPipe model, we need to divide the z coordinate by the height of the image.
        all_landmarks[:, 2] = all_landmarks[:, 2] / padded_face.shape[0]

        # # unlike in case of landmarks outputted by the actual mediapipe model, ours currently are already
        # # scaled by the image dimensions, so no need to perform line 96 in deconstruct-mediapipe/test_converted_model.py
        blendshape_input = all_landmarks[BLENDSHAPE_MODEL_LANDMARKS_SUBSET, :2]
        blendshape_input = blendshape_input.unsqueeze(0) # add batch dimension
        # with torch.no_grad(): # is this necessary or strictly not allowed?
        blendshapes = self.blendshape_model(blendshape_input)
        blendshapes = blendshapes.squeeze()
        return all_landmarks, blendshapes, padded_face


# NOTE: For eye blendshape scores to be truly reliable, we'd likely need to map the refined eye contour landmarks
# returned by the iris landmarker to those returned by the face mesh landmarker, and then replace them
# I think this is exactly the resource we need to do that: https://github.com/google-ai-edge/mediapipe/blob/ab1de4fced96c18b79eda4ab407b04eb08301ea4/mediapipe/graphs/iris_tracking/calculators/update_face_landmarks_calculator.cc#L250C3-L260C4

##########################################################################
#################### TESTING PyTorchMediapipeFaceLandmarker ####################
##########################################################################

mp = PyTorchMediapipeFaceLandmarker()
# validation on image
img_path = "data/obama2.jpeg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.tensor(img, dtype=torch.float32, requires_grad = True) # emulate format that will be output by generator
landmarks, blendshapes, padded_face = mp(img_tensor)

# vis and compare
padded_face = padded_face.detach().numpy().astype(np.uint8)
blendshapes_np = blendshapes.detach().numpy()
compare_to_real_mediapipe(landmarks, blendshapes_np, padded_face, save_landmark_comparison=True)
# W = torch.tensor(padded_face.shape[1])
# H = torch.tensor(padded_face.shape[0])
# aligned3d, aligned2d  = align_landmarks_differentiable(landmarks, W, H, W, H)

# # generate computation graph visualization for mediapipe facemesh 
# dot = make_dot(landmarks, params=dict(mp.named_parameters()))
# dot.format = 'png'
# dot.render('facemesh_graph')


# webcam live demo
# cap = cv2.VideoCapture(0)
# count = 0
# while True:
#     ret, img = cap.read()
#     if not ret:
#         break
#     count += 1
#     if count < 4:
#         continue
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     mtcnn = MTCNN()
#     bboxes, probs = mtcnn.detect(img)
#     if bboxes is None:
#         continue
#     bbox = bboxes[0]
#     bbox = bbox + [-50, -50, 50, 50] # add padding to the bbox, based on observation that mediapipe landmarks extractor benefits from this
#     x1, y1, x2, y2 = bbox.astype(int)
#     img = img[y1:y2, x1:x2, :]
#     img_tensor = torch.tensor(img, dtype=torch.float32, requires_grad = False)
#     landmarks, blendshapes, padded_face = mp(img_tensor)
#     padded_face = padded_face.detach().numpy().astype(np.uint8)
#     blendshapes_np = blendshapes.detach().numpy()
#     compare_to_real_mediapipe(landmarks, blendshapes_np, padded_face, live_demo = True)


