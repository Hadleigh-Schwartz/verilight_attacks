import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import mp_alignment

from signal_utils import single_feature_signal_processing, rolling_average
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import config

import sys
if sys.platform == 'linux':
    sys.path.append('/home/xiaofeng/deepfake_detection/system/e2e/common')
else:
    sys.path.append(f"{os.path.expanduser('~')}/deepfake_detection/system/e2e/common")
from ultralight_face import UltraLightFaceDetector

if sys.platform == 'linux':
    sys.path.append('/home/xiaofeng/deepfake_detection/system/e2e/visualization')
else:
    sys.path.append(f"{os.path.expanduser('~')}/deepfake_detection/system/e2e/visualization")
from feature_vis import annotate_frame, resize_img
from mediapipe_vis import draw_landmarks_on_image


def create_dynamic_hash_from_dynamic_features(dynamic_features):
    
    #make features into signal(s)
    signals = [[] for i in range(len(config.target_features))]
    for frame_feats in dynamic_features:
        for i in range(len(config.target_features)):
            signals[i].append(frame_feats[i])
    # print("Len signal", len(signals[0]))
    #interp nans of signal(s), downsample to fixed number of frames per window, and create final concatenated signal
    concat_processed_signal = []
    proc_signals = [] # ultimately for visualization purposes
    for signal in signals:
        proc_signal = single_feature_signal_processing(signal)
        # print(f"Signal length: {len(proc_signal)}")
        proc_signals.append(proc_signal)
        concat_processed_signal += proc_signal
    concat_processed_signal = np.array(concat_processed_signal, dtype=np.float64)
    concat_processed_signal -= concat_processed_signal.mean()   #must zero mean it so that the Pearson correlation equals the cosine similarity

    return signals, proc_signals, concat_processed_signal

    # dynamic_feat_hash, signals, proc_signals, concat_processed_signal = create_dynamic_hash_from_dynamic_features(dynamic_features, config.dynamic_fam, config.dynamic_hash_funcs)
        
class MPExtractor(object):
    def __init__(self):
        #set up initial face detector, if using
        if config.intitial_face_detection == True:
            digest_extraction_log("Initializing face detector", "INFO")
            #self.face_detector = RetinaFaceDetector("mobile0.25") #other option: "resnet50"
            #sys.stdout = open(os.devnull, "w")
            if sys.platform == 'linux':
                self.face_detector = UltraLightFaceDetector("slim", "cuda", 0.7)
            else:
                self.face_detector = UltraLightFaceDetector("slim", "mps", 0.7)
            #sys.stdout = sys.__stdout__
            digest_extraction_log("Done initializing face detector", "INFO")
    
        #set up MediaPipe
        digest_extraction_log("Setting up MediaPipe FaceMesh", "INFO")
        # sys.stdout = open(os.devnull, "w")
        # sys.stderr = open(os.devnull, "w")
        if sys.platform == 'linux':
            base_options = python.BaseOptions(model_asset_path='/home/hadleigh/deepfake_detection/system/dev/feature_extraction/face_landmarker_v2_with_blendshapes.task')
        else:
            base_options = python.BaseOptions(model_asset_path=f"{os.path.expanduser('~')}/deepfake_detection/system/dev/feature_extraction/face_landmarker_v2_with_blendshapes.task")
        
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            num_faces=1, 
                                            min_face_detection_confidence=.25, 
                                            min_face_presence_confidence=.25, 
                                            min_tracking_confidence=.25,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,)
        self.extractor = vision.FaceLandmarker.create_from_options(options)
        # sys.stdout = sys.__stdout__
        # sys.stderr = sys.__stderr__
        digest_extraction_log("Done setting up MediaPipe FaceMesh'", "INFO")
        
    def get_pair_dist(self, coord1, coord2, bbox = None):
        """
        Note to selves: 2D and 3D distances are the same except for difference of scale!
        So their trends are exactly the same.
        """

        x_diff = coord1[0] - coord2[0]
        y_diff = coord1[1] - coord2[1]

        if bbox is not None:
            bbox_W = bbox[1][0] - bbox[0][0]
            bbox_H = bbox[1][1] - bbox[0][1]
            x_diff /= bbox_W
            y_diff /= bbox_H
    
        dist = np.sqrt(x_diff**2 + y_diff**2) 

        return dist
            
    def get_mp_bbox(self, coords):
        """
        Get face bounding box coordinates for a frame with frame index based on MediaPipe's extracted landmarks 

        Parameters
        ----------
        coords : list of 2D tuples
            2D facial landmarks
        """
        cx_min = float('inf')
        cy_min = float('inf')
        cx_max = cy_max = 0
        for coord in coords:
            cx, cy = coord
            if cx < cx_min:
                cx_min = cx
            if cy < cy_min:
                cy_min = cy
            if cx>cx_max:
                cx_max = cx
            if cy > cy_max:
                cy_max = cy
        bbox = [(cx_min, cy_min), (cx_max, cy_max)]
        return bbox

    def extract_features(self, frame, frame_num = None):
        input_frame_H, input_frame_W, _ = frame.shape
        if config.intitial_face_detection:
            #run initial face detection
            initial_face_bbox = self.face_detector.detect(frame)
            if len(initial_face_bbox) == 0:
                frame = None
            else:
                # get crop of frame to pass to facial landmark extraction
                bottom = max(initial_face_bbox[1] - config.initial_bbox_padding, 0)
                top = min(initial_face_bbox[3]+1 + config.initial_bbox_padding, input_frame_H)
                left = max( initial_face_bbox[0] - config.initial_bbox_padding, 0)
                right = min(initial_face_bbox[2] + 1 + config.initial_bbox_padding, input_frame_W)
                frame = frame[bottom:top,left:right]
        else:
            initial_face_bbox = None
        
      
        if frame is None: 
            #if no face was detected with initial detection, return nans
            feat_vals = [np.nan for i in range(len(config.target_features))]
            return feat_vals, None, None
        else:
            #run facial landmark detection 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = self.extractor.detect(mp_img)  
            face_landmarks_list = detection_result.face_landmarks
            if len(face_landmarks_list) == 0:
                feat_vals = [np.nan for i in range(len(config.target_features))]
                return feat_vals, None, None

            face_landmarks = face_landmarks_list[0] 
            H, W, _ = frame.shape #not the same as self.input_H, self.input_W if initial face detection (and thus cropping) is being used!
            # MediaPipe by deafult returns facial landmark coordinates normalized to 0-1 based on the input frame dimensions. Here we 
            # un-normalize to get coordinates ranging from 0-W/0_H (i.e., actual pixel coordinates)
            landmark_coords = [(landmark.x * W, landmark.y * H, landmark.z) for landmark in face_landmarks] 
            _, landmark_coords_2d_aligned  = mp_alignment.align_landmarks(landmark_coords, input_frame_W, input_frame_H, W, H)
            blendshapes = detection_result.face_blendshapes[0]

            if config.bbox_norm_dists:
                bbox = self.get_mp_bbox(landmark_coords_2d_aligned)

            feat_vals = []
            for feat in config.target_features:
                if type(feat) == int: #blendshape
                    feat_vals.append(blendshapes[feat].score)
                else: #dist
                    landmark1_coord, landmark2_coord = landmark_coords_2d_aligned[int(feat.split("-")[0])], landmark_coords_2d_aligned[int(feat.split("-")[1])]
                    d = self.get_pair_dist(landmark1_coord, landmark2_coord, bbox)
                    feat_vals.append(d)
            
            return feat_vals, initial_face_bbox, detection_result
        

class VideoDigestExtractor(object):
    """
    Class for extracting digests from a video, i.e., offline verification of a video's integrity.
    """
    def __init__(self, video_path):
        self.mp_extractor = MPExtractor()
        self.video_path = video_path
 
    
    def extract_from_video_slice(self, start_frame, end_frame, seq_num, vis_output_path = None):
        """
        Given the start and end frame numbers of the target window <seq_num> within the video at <video_path>,
        extract the digest for this window
        """
 
        cap = cv2.VideoCapture(self.video_path)
        frame_num = 0
        dynamic_features = []
        # START FOR DEMO VIDEO CONTENT
        # plain_frames = []
        # END FOR DEMO VIDEO CONTENT
        if vis_output_path:
            annotated_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if not (frame_num >= start_frame):
                frame_num += 1
                continue
            if frame_num > end_frame:
                break
            if frame_num == start_frame:
                identity_features = self.id_extractor.extract(frame)
            frame_feats, face_bbox, detection_result = self.mp_extractor.extract_features(frame)
            dynamic_features.append(frame_feats)
            frame_num += 1

            # START FOR DEMO VIDEO CONTENT
            # plain_frames.append(frame)
            # END FOR DEMO VIDEO CONTENT

            if vis_output_path:
                frame = annotate_frame(frame, face_bbox, detection_result)    
                annotated_frames.append(frame)

        digest_payload, proc_signals, concat_processed_signal = create_digest_from_features(dynamic_features, identity_features, seq_num)

        # with open(f"ug{seq_num}.pkl", "wb") as pklfile:
        #     pickle.dump(proc_signals, pklfile)
        #     pickle.dump(concat_processed_signal, pklfile)
        #     pickle.dump(dynamic_features, pklfile)
        
        if vis_output_path:
            self.gen_vis_video(annotated_frames, seq_num, dynamic_features, proc_signals, concat_processed_signal, vis_output_path)

        # START FOR PAPER PLOTTING
        # self.save_raw_signals(dynamic_features)
        # self.save_proc_signals(proc_signals)
        # END FOR PAPER PLOTTING

        # START FOR DEMO VIDEO CONTENT
        # self.gen_demo_video(plain_frames, dynamic_features, proc_signals, include_raw_frames=True)
        # END FOR DEMO VIDEO CONTENT

        return digest_payload, proc_signals, concat_processed_signal

    def save_proc_signals(self, proc_signals, output_path = "./"):
        vid_name = self.video_path.split("/")[-1].split(".")[0]
        for f, s in enumerate(proc_signals):
            file = open(f"{vid_name}_proc_signal_{config.target_features[f]}.csv", "w")
            file.write("frame_num,signal_val\n")
            for i, val in enumerate(s):
                file.write(f"{i},{val}\n")
    
    def save_raw_signals(self, dynamic_features, output_path = "./"):
        vid_name = self.video_path.split("/")[-1].split(".")[0]
        signals = [[] for i in range(len(config.target_features))]
        # make raw dynamic features list into a list of signals, one for each
        for frame_feats in dynamic_features:
            for i in range(len(config.target_features)):
                signals[i].append(frame_feats[i])
        for f, s in enumerate(signals):
            file = open(f"{vid_name}_raw_signal_{config.target_features[f]}.csv", "w")
            file.write("frame_num,signal_val\n")
            for i, val in enumerate(s):
                file.write(f"{i},{val}\n")

    def gen_demo_video(self, frames, dynamic_features, proc_signals, output_path = "demo_content.mp4", signal_type = "raw", include_raw_frames = False):
        colors = ['#377eb8', "green", "cyan", "red", "orchid", "darkorchid", "crimson", "lime", "fuchsia", "#ff7f00", "#f781bf", "darkcyan", "yellowgreen", "#4daf4a", "cornflowerblue",  "peru"]

        blendshape_names = [
            "_neutral",
            "browDownLeft",
            "browDownRight",
            "browInnerUp",
            "browOuterUpLeft",
            "browOuterUpRight",
            "cheekPuff",
            "cheekSquintLeft",
            "cheekSquintRight",
            "eyeBlinkLeft",
            "eyeBlinkRight",
            "eyeLookDownLeft",
            "eyeLookDownRight",
            "eyeLookInLeft",
            "eyeLookInRight",
            "eyeLookOutLeft",
            "eyeLookOutRight",
            "eyeLookUpLeft",
            "eyeLookUpRight",
            "eyeSquintLeft",
            "eyeSquintRight",
            "eyeWideLeft",
            "eyeWideRight",
            "jawForward",
            "jawLeft",
            "jawOpen",
            "jawRight",
            "mouthClose",
            "mouthDimpleLeft",
            "mouthDimpleRight",
            "mouthFrownLeft",
            "mouthFrownRight",
            "mouthFunnel",
            "mouthLeft",
            "mouthLowerDownLeft",
            "mouthLowerDownRight",
            "mouthPressLeft",
            "mouthPressRight",
            "mouthPucker",
            "mouthRight",
            "mouthRollLower",
            "mouthRollUpper",
            "mouthShrugLower",
            "mouthShrugUpper",
            "mouthSmileLeft",
            "mouthSmileRight",
            "mouthStretchLeft",
            "mouthStretchRight",
            "mouthUpperUpLeft",
            "mouthUpperUpRight",
            "noseSneerLeft",
            "noseSneerRight"
        ]

        # width = 1200
        # height = 600
        width = 1200
        height = 240
        frame_shape =  (frames[0].shape[1], frames[0].shape[0])
        scaled_frame_shape = (int(frame_shape[0]*height/frame_shape[1]), height)
        final_output_width = width + scaled_frame_shape[0]
        if include_raw_frames:
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), 30, (final_output_width, height))
        else:
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), 30, (width, height))

        if signal_type == "raw":
            raw_signals = [[] for i in range(len(config.target_features))]
            max_sig_val = 0
            min_sig_val = 10000
            # make raw dynamic features list into a list of signals, one for each
            for feat_num, frame_feats in enumerate(dynamic_features):
                for i in range(len(config.target_features)):
                    raw_signals[i].append(frame_feats[i])
                    if frame_feats[i] > max_sig_val:
                        max_sig_val = frame_feats[i]
                    if frame_feats[i] < min_sig_val:
                        min_sig_val = frame_feats[i]
            #process with rolling average 
            signals = []
            for r in raw_signals:
                scaler = MinMaxScaler()
                r = np.array(r)
                mask = np.isnan(r)
                r[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), r[~mask]) # fill nans. mostly imporatnt for 60 far, which actually has Nans
                r = scaler.fit_transform(r.reshape(-1, 1)).reshape(-1) #normalize for visualization
                r = rolling_average(r, n = 2)
                signals.append(r)
        else:
            signals = proc_signals
            max_sig_val = np.array(proc_signals).max()
            min_sig_val = np.array(proc_signals).min()
        
        max_sig_val = 1
        min_sig_val = 0
        # target_vis = [0, 9, 13]
        target_vis = [5, 6, 7, 8, 9]
        for i in range(len(frames)):
            for f, s in enumerate(signals):
                if f not in target_vis:
                    continue
                if type(config.target_features[f]) == int:
                    label = blendshape_names[config.target_features[f]]
                else:
                    label = config.target_features[f]
                
                plt.plot(s[:i + 1], label = label, c = colors[f])
            
            if i < 50:
                xrange = (0, 100)
            else:
                xrange = (i - 50, i + 50)
            plt.xlim(xrange)
            plt.ylim((min_sig_val - min_sig_val*.1, max_sig_val))
            # remove ylavels and tics
            # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            # plt.margins(0,0)
            # plt.gca().axes.get_yaxis().set_visible(False)
            # plt.gca().axes.get_xaxis().set_visible(False)
            leg = plt.legend(loc='upper right', fontsize=(20))
            for line in leg.get_lines():
                line.set_linewidth(4.0)
            # leg.get_texts()[0].set_text('Lip Distance')
            # leg.get_texts()[1].set_text('Eye Blendshape')
            # leg.get_texts()[2].set_text('Jaw Blendshape')
            figure = plt.gcf()
            figure.set_dpi(100)
            figure.set_size_inches(0.01*width, 0.01*height)
            figure.canvas.draw()
            fig_img = np.array(figure.canvas.buffer_rgba())
            fig_img = cv2.cvtColor(fig_img, cv2.COLOR_RGBA2BGR)
            plt.clf()

            if include_raw_frames:
                # draw landmarked frame
                frame = resize_img(frames[i], scaled_frame_shape[0], scaled_frame_shape[1]) # resize frame
                # if frame.shape[0] != fig_img.shape[0]:
                #     print("Alert frame shape mismatch")
                out_frame = np.hstack((frame, fig_img))
                out.write(out_frame)
            else:
                out.write(fig_img)
          
            # fig_img = resize_img(fig_img, width*2, height*2)
            # out_vid_frame = np.hstack((frame, fig_img))

        out.release()

    def gen_vis_video(self, annotated_frames, seq_num, dynamic_features, proc_signals, concat_processed_signal, output_path):
        height = 1000
        width = 1000
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), 30, (width * 4, height*2))
  
        max_sig_val = 0
        min_sig_val = 10000
        signals = [[] for i in range(len(config.target_features))]
        # make raw dynamic features list into a list of signals, one for each
        for frame_feats in dynamic_features:
            for i in range(len(config.target_features)):
                signals[i].append(frame_feats[i])
                if frame_feats[i] > max_sig_val:
                    max_sig_val = frame_feats[i]
                if frame_feats[i] < min_sig_val:
                    min_sig_val = frame_feats[i]
        
        for i in range(len(annotated_frames)):
            # print("annotating ", i)
            frame = annotated_frames[i]
            frame = resize_img(frame, width*2, height*2)

            for f, s in enumerate(signals):
                plt.plot(s[:i + 1], label = config.target_features[f])
            
            plt.xlim(0, len(annotated_frames))
            #plt.ylim(min(concat_processed_signal), max(concat_processed_signal))
            plt.ylim((min_sig_val, max_sig_val))
            figure = plt.gcf()
            plt.title(f"Seq {seq_num} Feature Signals (Raw)")
            plt.legend(loc='upper right')
            figure.set_dpi(100)
            figure.set_size_inches(0.01*width*2, 0.01*height)
            figure.canvas.draw()
            fig_img = np.array(figure.canvas.buffer_rgba())
            fig_img = cv2.cvtColor(fig_img, cv2.COLOR_RGBA2BGR)
            plt.clf()
            fig_img = resize_img(fig_img, width*2, height*2)
            out_vid_frame = np.hstack((frame, fig_img))
            out.write(out_vid_frame)
        
        # conclude with a summary frame with the original, downsampled, and concatenated signals
        for f, s in enumerate(proc_signals):
            plt.plot(s, label = config.target_features[f])

        plt.xlim(0, len(annotated_frames))
        plt.ylim(min(concat_processed_signal), max(concat_processed_signal))
        figure = plt.gcf()
        plt.legend(loc='upper right')
        plt.title(f"Seq {seq_num} Feature Signals (Processed)")
        figure.set_dpi(100)
        figure.set_size_inches(0.01*width*2, 0.01*height)
        figure.canvas.draw()
        top_fig_img = np.array(figure.canvas.buffer_rgba())
        top_fig_img = cv2.cvtColor(top_fig_img, cv2.COLOR_RGBA2BGR)
        plt.clf()

        empty = np.zeros((height, width*2, 3)).astype(np.uint8)
        top_row = np.hstack((empty, top_fig_img))

        # get resampled from concatenated signal
        feat_count = 0
        for f in range(0, len(concat_processed_signal), config.single_dynamic_signal_len):
            plt.plot(concat_processed_signal[f:f+config.single_dynamic_signal_len], label = f"{config.target_features[feat_count]}")
            feat_count += 1
        figure = plt.gcf()
        plt.legend(loc='upper right')
        plt.title(f"Seq {seq_num} Resampled Feature Signals ({config.target_samples_per_second} sps)")
        figure.set_dpi(100)
        figure.set_size_inches(0.01*width*2, 0.01*height)
        figure.canvas.draw()
        bottom_fig_img = np.array(figure.canvas.buffer_rgba())
        bottom_fig_img = cv2.cvtColor(bottom_fig_img, cv2.COLOR_RGBA2BGR)
        plt.clf()

        plt.plot(concat_processed_signal)
        plt.vlines([f for f in range(0, len(concat_processed_signal), config.single_dynamic_signal_len)], min(concat_processed_signal), max(concat_processed_signal), linestyles='dashdot', colors='r', label="Individual feature starts")
        plt.legend(loc = 'upper right')
        plt.title(f"Seq {seq_num} Concatenated Resampled Feature Signals")
        figure = plt.gcf()
        figure.set_dpi(100)
        figure.set_size_inches(0.01*width*2, 0.01*height)
        figure.canvas.draw()
        concat_fig_img = np.array(figure.canvas.buffer_rgba())
        concat_fig_img = cv2.cvtColor(concat_fig_img, cv2.COLOR_RGBA2BGR)
        plt.clf()

        bottom_row = np.hstack((concat_fig_img, bottom_fig_img))

        final = np.vstack((top_row, bottom_row))
        
        for q in range(90):
            out.write(final)

        out.release()
    
