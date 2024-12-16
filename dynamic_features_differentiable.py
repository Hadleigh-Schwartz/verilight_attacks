
import sys
import cv2
import torch
from torch import nn
import numpy as np
from hadleigh_utils import pad_image, get_real_mediapipe_results, compare_to_real_mediapipe, record_face_video, compute_method_differences
from mp_alignment_differentiable import align_landmarks as align_landmarks_differentiable
from mp_alignment_original import align_landmarks as align_landmarks_original
from torchviz import make_dot
from matplotlib import pyplot as plt
from mp_face_landmarker import PyTorchMediapipeFaceLandmarker

sys.path.append("facenet-pytorch")
from models.mtcnn import MTCNN

def visualize_signals(signals, padded_faces, landmarks_over_time):
    target_features = [(0, 17), (40, 17), (270, 17), (0, 91), (0, 321),
                                 6, 7, 8, 9, 10, 11, 12, 23, 25, 50, 51] 
    vis_feature_ids = [5, 6, 7, 8, 9]
    signals_min = signals.min()
    signals_max = signals.max()
    out = None
    for f in range(len(padded_faces)):
        fig, ax = plt.subplots()
        for i in range(signals.shape[0]):
            if i not in vis_feature_ids:
                continue
            signal = signals[i, :f]
            ax.plot(signal, label = f"{target_features[i]}")
        ax.set_xlim(max(0, f - 25), min(f + 25, len(padded_faces)))
        ax.set_ylim(signals_min, signals_max)
        ax.legend()
        # get plot as image
        fig = plt.gcf()
        fig.canvas.draw()
        fig_arr = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        fig_arr = fig_arr[:, :, ::-1]

        # draw landmarks on the image
        frame = padded_faces[f]
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_landmarks = landmarks_over_time[f]
        for t in target_features:
            if type(t) != int:
                lm1 =  tuple(frame_landmarks[t[0]][:2])
                lm2 = tuple(frame_landmarks[t[1]][:2])
                lm1 = (int(lm1[0]), int(lm1[1]))
                lm2 = (int(lm2[0]), int(lm2[1]))
                cv2.circle(frame, lm1, 1, (0, 255, 0), -1)
                cv2.circle(frame, lm2, 1, (0, 255, 0), -1)
                cv2.line(frame,lm1, lm2, (0, 255, 0), 1)

        
        frame_shape = frame.shape
        arr_shape = fig_arr.shape
        if frame_shape[0] < arr_shape[0]:
            # add padding to bottom
            padding = np.zeros((arr_shape[0] - frame_shape[0], frame_shape[1], frame_shape[2]), dtype=np.uint8)
            frame = np.concatenate([frame, padding], axis=0)
        elif frame_shape[0] > arr_shape[0]:
            padding = np.zeros((frame_shape[0] - arr_shape[0], arr_shape[1], arr_shape[2]), dtype=np.uint8)
            fig_arr = np.concatenate([fig_arr, padding], axis=0)
        vis = np.concatenate([frame, fig_arr], axis=1)
        if out is None:
            out = cv2.VideoWriter("signals_vis.mp4", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 20, (vis.shape[1], vis.shape[0]))
        out.write(vis)
    out.release()
       

def compute_perframe_method_differences(landmarks_over_time, blendshapes_over_time, padded_faces):

    for i in range(len(padded_faces)):
        landmarks = landmarks_over_time[i]
        blendshapes = blendshapes_over_time[i]
        padded_face = padded_faces[i]
        landmarks_diff, blendshapes_diff = compute_method_differences(landmarks, blendshapes, padded_face)


class VeriLightDynamicFeatures(nn.Module):
    def __init__(self):
        super(VeriLightDynamicFeatures, self).__init__()

        self.mp = PyTorchMediapipeFaceLandmarker()

        self.target_features = [(0, 17), (40, 17), (270, 17), (0, 91), (0, 321),
                                 6, 7, 8, 9, 10, 11, 12, 23, 25, 50, 51] 
    
    def get_mp_bbox(self, coords):
        """
        Get face bounding box coordinates for a frame with frame index based on MediaPipe's extracted landmarks 

        Parameters
        ----------
        coords : list of 2D tuples
            2D facial landmarks
        """
        cx_min = torch.min(coords[:, 0])
        cy_min = torch.min(coords[:, 1])
        cx_max = torch.max(coords[:, 0])
        cy_max = torch.max(coords[:, 1])
        bbox = torch.tensor([[cx_min, cy_min], [cx_max, cy_max]])
        return bbox
       
    def forward(self, video_tensor):
        """
        IMPORTANT: All videos underlying the video_tensor should have the same framerate and duration.
        """
        padded_faces = [] # for vis only
        landmarks_over_time = []# for vis only
        blendshapes_over_time = [] # for different computation only
        feature_values = torch.empty(video_tensor.shape[0], len(self.target_features)) # list of values for each feature in self.target_features for each frame
        for i in range(video_tensor.shape[0]):
            frame = video_tensor[i, :, :, :]
            landmarks, blendshapes, padded_face = self.mp(frame)

            landmarks_curr = landmarks.detach().numpy().copy()# for vis. idk why but if i dont do copy it takes on the value of the lanmdarks noramlized in alignment...
            landmarks_over_time.append(landmarks_curr)
            padded_faces.append(padded_face.detach().numpy()) # for vis 
            blendshapes_over_time.append(blendshapes.detach().numpy())

            if torch.all(landmarks == 0): # no face detected
                if i != 0: #repeat previous row, mimicing a kind of interpolation
                    feature_values[i, :] = feature_values[i-1, :]
                else: # unfortuantely this is the first frame so we have nothing to repeat.
                    # to be safe just put zeros
                    for feat_num, feature in enumerate(self.target_features):
                        feature_values[i,feat_num] = 0
                continue

            W, H = torch.tensor(padded_face.shape[1]), torch.tensor(padded_face.shape[0])
            _, landmark_coords_2d_aligned = align_landmarks_differentiable(landmarks, W, H, W, H)

            # # draw landmarks on blank
            # blank = np.ones((H, W, 3), dtype=np.uint8)*255
            # for l in landmark_coords_2d_aligned:
            #     x, y = l[:2].detach().numpy().astype(np.int8)
            #     cv2.circle(blank, (x, y), 1, (0, 255, 0), -1)
            # cv2.imshow("Aligned Landmarks", blank)
            # cv2.waitKey(0)

            bbox = self.get_mp_bbox(landmark_coords_2d_aligned)
            bbox_W = bbox[1, 0] - bbox[0, 0]
            bbox_H = bbox[1, 1] - bbox[0, 1]
            for feat_num, feature in enumerate(self.target_features):
                if type(feature) == int: # this is a blendshape
                    feature_values[i,feat_num] = blendshapes[feature]
                else: # this is a facial landmark distance
                    lm1 = landmark_coords_2d_aligned[feature[0]]
                    lm2 = landmark_coords_2d_aligned[feature[1]]
                    x_diff = lm1[0] - lm2[0]
                    x_diff /= bbox_W
                    y_diff = lm1[1] - lm2[1]
                    y_diff /= bbox_H
                    distance = torch.sqrt(x_diff**2 + y_diff**2)
                    feature_values[i,feat_num] = distance

        # now we have a tensor of shape (frames, features) 
        # where each row is the feature values for a frame
        # transposing gives us a tensor of shape (features, frames)
        # where each row is the featrure's signal over time
        signals = feature_values.T
        
        """
        process the signals in the same fashion VeriLight does in real-time/during verifcation. This entails
        1. interpolating to fill missing values, which result from frames where no face/landmarks is detected
        2. standard scaling each feature's signal with respect to itself
        3. applying a rolling average to each feature's signal to smooth inter-frame jitter
        4. resample the signals to a fixed length, to account for potential varying video framerates
        5. concatenating all the signals into a single feature vector
        6. zero-meaning the feature vector
        we can't do the interpolation used in orignial code in a differentiable manner
        , so we've approxcimately taken care of that above in the loop with the repetitions

        We can't directly achieve (1), because it wouldn't be differentiable, so we've approximated it by repeating the previous row's values.
        We also can't directly achieve (4) for the same reason, but that's ok because we can skip it here: we will enforce during data processing
        that all videos have the same framerate and duration, leading to signals of the same length
        """
        # standard scale each row
        signals = (signals - signals.mean(dim=1, keepdim=True)) / signals.std(dim=1, keepdim=True)
    
        # apply rolling average to each row
        # https://discuss.pytorch.org/t/doing-a-very-basic-simple-moving-average/5003/6
        kernel = 2
        sma = nn.AvgPool1d(kernel_size=kernel, stride = 1)
        smoothed_signals = sma(signals)
        signal_len_diff =  signals.shape[1] - smoothed_signals.shape[1]
        side_reps = signal_len_diff // 2
        front_val = smoothed_signals[:, 0].unsqueeze(1).repeat(1, side_reps)
        back_val = smoothed_signals[:, -1].unsqueeze(1).repeat(1, signal_len_diff - side_reps)
        smoothed_signals = torch.cat([front_val, smoothed_signals, back_val], dim=1)

        # visualize_signals(smoothed_signals.detach().numpy(), padded_faces, landmarks_over_time) # visualization has to be done on padded faces because that's what landmarks are extracted on
        # compute_perframe_method_differences(landmarks_over_time, blendshapes_over_time, padded_faces)

        # concatenate all the signals into a single feature vector
        dynamic_feature_vec = smoothed_signals.view(-1)
        assert torch.all(dynamic_feature_vec[:video_tensor.shape[0]] == smoothed_signals[0, :video_tensor.shape[0]])
        assert torch.all(dynamic_feature_vec[14*video_tensor.shape[0]:15*video_tensor.shape[0]] == smoothed_signals[14, :video_tensor.shape[0]])

        # zero mean the feature vector
        dynamic_feature_vec = (dynamic_feature_vec - dynamic_feature_vec.mean()) 


        return dynamic_feature_vec
            

