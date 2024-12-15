
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
from mp_face_landmarker import PyTorchMediapipeFaceLandmarker

sys.path.append("facenet-pytorch")
from models.mtcnn import MTCNN



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
        feature_values = torch.empty(video_tensor.shape[0], len(self.target_features)) # list of values for each feature in self.target_features for each frame
        for i in range(video_tensor.shape[0]):
            frame = video_tensor[i, :, :, :]
            landmarks, blendshapes, padded_face = self.mp(frame)

            landmarks_curr = landmarks.detach().numpy()# for vis. idk why but if i dont do copy it takes on the value of the lanmdarks noramlized in alignment...
            np.save(f"landmarks_frame{i}.npy", landmarks_curr)
            padded_faces.append(padded_face.detach().numpy()) # for vis 

            if torch.all(landmarks == 0): # no face detected
                print("NO FACE DETECTE, VL")
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
        
        # process the signals. 
        # we can't do the interpolation used in orignial code in a differentiable manner
        #, so we've approxcimately taken care of that above in the loop with the repetitions
        # standard scale each row
        signals = (signals - signals.mean(dim=1, keepdim=True)) / signals.std(dim=1, keepdim=True)

        # save the signals, padded faces, and landmarks over time
        np.save("signals.npy", signals.detach().numpy())
        np.save("padded_faces.npy", padded_faces)
        
        # visualize_signals(signals, landmarks_over_time, padded_faces) # visualization has to be done on padded faces because that's what landmarks are extracted on

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

        # no need to resample because we will enforce during data processing
        # that all videos have the same framerate and duration, leading to signals of the same length

        # concatenate all the signals into a single feature vector
        dynamic_feature_vec = smoothed_signals.view(-1)
        assert torch.all(dynamic_feature_vec[:video_tensor.shape[0]] == smoothed_signals[0, :video_tensor.shape[0]])
        assert torch.all(dynamic_feature_vec[14*video_tensor.shape[0]:15*video_tensor.shape[0]] == smoothed_signals[14, :video_tensor.shape[0]])

        # zero mean the feature vector
        dynamic_feature_vec = (dynamic_feature_vec - dynamic_feature_vec.mean()) 


        print(dynamic_feature_vec.shape, dynamic_feature_vec.grad_fn)
        
        return dynamic_feature_vec
            



vl = VeriLightDynamicFeatures()
# validation on video
cap = cv2.VideoCapture("data/test_video.mp4")
# read video in as a tensor of shape (frames, H, W, 3), RGB order
frames = []
max_frames = 300 # set for vis/testing 
frame_num = 0
while True:
    ret, img = cap.read()
    if not ret:
        break 
    frame_num += 1
    if frame_num > max_frames:
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img, dtype=torch.float32, requires_grad = True)
    frames.append(img_tensor)
frames = torch.stack(frames)
test = vl(frames)

# generate computation graph visualization for verilight dynamic feature extractror
# sys.setrecursionlimit(10000) # to avoid RecursionError when building the graph
# dot = make_dot(test, params=dict(vl.named_parameters()))
# dot.format = 'png'
# dot.render('dyn_feat_graph')


def visualize_signals():
    target_features = [(0, 17), (40, 17), (270, 17), (0, 91), (0, 321),
                                 6, 7, 8, 9, 10, 11, 12, 23, 25, 50, 51] 
    signals = np.load("signals.npy")
    padded_faces = np.load("padded_faces.npy", allow_pickle=True)
    signals_min = signals.min()
    signals_max = signals.max()
    out = None
    for f in range(len(padded_faces)):
        fig, ax = plt.subplots()
        for i in range(signals.shape[0]):
            if not (i > 5 and i < 10):
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
        frame_landmarks = np.load(f"landmarks_frame{f}.npy")
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
       
            
visualize_signals()
