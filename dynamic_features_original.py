import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import crc8

from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


import mp_alignment

from signal_utils import single_feature_signal_processing, rolling_average
from rp_lsh import hash_point
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import config
from bitstring_utils import pad_bitstring, bitstring_to_bytes, bytes_to_bitstring

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

def digest_extraction_log(message, log_level):
    """
    Logging function for the digest extraction process.

    Parameters:
        message (str): The message to log
        log_level (str): The log level of the message. Can be "DEBUG", "INFO", "WARNING", or "ERROR"
    
    Returns:
        None
    """
    if log_level == "DEBUG":
        if config.LOG_LEVEL == "DEBUG":
            print("FEATURE EXTRACTOR [DEBUG]: {}".format(message))
    elif log_level == "INFO":
        if config.LOG_LEVEL == "DEBUG" or config.LOG_LEVEL == "INFO":
            print("FEATURE EXTRACTOR [INFO]: {}".format(message))
    elif log_level == "WARNING":
        if config.LOG_LEVEL == "DEBUG" or config.LOG_LEVEL == "INFO" or config.LOG_LEVEL == "WARNING":
            print("FEATURE EXTRACTOR [WARNING]: {}".format(message))
    elif log_level == "ERROR":
        print("FEATURE EXTRACTOR[ERROR]: {}".format(message))


def create_dynamic_hash_from_dynamic_features(dynamic_features, dynamic_hash_fam, dynamic_hash_funcs):
    """
    Creates our dynamic hash from the dynamic features. This involves converting the dynamic features into a signal and then 
    applying the locality sensitive hashing.

    Parameters:
        dynamic_features (list): List of N lists, where N is the number of frames. Each of the N elements is 
        itself a list, where each element is the value of one feature. For example, the dynamic features for
        3 frames, using 5 blendshapes/distances, could something like
        [[0.02, 0.5, 0.6, 0.03, 0.2], [0.02, 0.3, 0.3, 0.03, 0.18], [0.02, 0.52, 0.4, 0.06, 0.2]]

        dynamic_hash_fam (CosineHashFamily object): The LSH family object used to hash the dynamic features. See rp_lsh.py for more details.
        dynamic_hash_funcs (list): The random projection functions used to hash the dynamic features. See rp_lsh.py for more details.
    
    Returns:
        dynamic_feat_hash (str): The hash of the dynamic features
        signals (list): The signals for each feature
        proc_signals (list): The processed signals for each feature
        concat_processed_signal (list): The concatenated processed signal (i.e., concatenation of all processed signals, zero meaned)
    """
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

    if np.count_nonzero(concat_processed_signal) == 0: #if signal is all zeros, return random bitstream "cover traffic" as the dyanmic hash, note this in log
        dynamic_feat_hash = pad_bitstring(format(random.getrandbits(config.dynamic_hash_k), '0b'), config.dynamic_hash_k)
    else:
        dynamic_feat_hash = hash_point(dynamic_hash_fam, dynamic_hash_funcs, concat_processed_signal)
    
    return dynamic_feat_hash, signals, proc_signals, concat_processed_signal

def create_digest_from_features(dynamic_features, identity_features, feature_seq_num, output_path = None, img_nums = None):
    """
    Given dynamic features, identity features, and feature_seq_num, returns the raw bits making up the digest (i.e., digest payload)
    that is embedded into the video.
    Specifically, this includes the feature seq num, concatenated dynamic feature signal hash and identity feature hash. 
    Optionally dumps the hashes, intermediate signals, and img_nums to a pickle at output_path.
    The parameters and LSH families used for hashing are specified in the config file. It's important tha the same LSH families
    used during the live embedding are used for verification.

    NOTE: Re CRC placement. In the paper we state that the checksum is added at the end of the coded signature. This is because, in practice, it would probably be
    good to have the unit ID and date ordinal also verified by the checksum, and not to even bother decrpyting the signature if the checksum is wrong.
    In this implementation, however, the checksum is added to the end of the digest, which isn't encrypted yet. 
    This is a slight error, but it doesn't change the results, since we don't actually use the unit ID and date ordinal in our verification process.

    Parameters:
        dynamic_features (list): List of N lists, where N is the number of frames. Each of the N elements is 
        itself a list, where each element is the value of one feature. For example, the dynamic features for
        3 frames, using 5 blendshapes/distances, could something like
        [[0.02, 0.5, 0.6, 0.03, 0.2], [0.02, 0.3, 0.3, 0.03, 0.18], [0.02, 0.52, 0.4, 0.06, 0.2]]

        identity_features (numpy array): the 512-dimensional ArcFace embedding

        output_path (str): Path to save pickle of features/signals to, if desired

        img_nums (list): List of image numbers corresponding to each frame in dynamic_features. Used for visualization purposes.
    
    Returns:
        payload (str): The bits, as string of '0' and '1' making up the digest payload
        proc_signals (list): The processed signals for each feature
        concat_processed_signal (list): The concatenated processed signal (i.e., concatenation of all processed signals, zero meaned)
    """

    #hash the dynamic feature signal and identity feature embedding, if possible
    dynamic_feat_hash, signals, proc_signals, concat_processed_signal = create_dynamic_hash_from_dynamic_features(dynamic_features, config.dynamic_fam, config.dynamic_hash_funcs)
    if np.count_nonzero(identity_features) == 0:
        id_feat_hash = pad_bitstring(format(random.getrandbits(config.identity_hash_k), '0b'), config.identity_hash_k) #if signal is all zeros, return random bitstream "cover traffic" as the dyanmic hash, note this in log
    else:
        id_feat_hash = hash_point(config.id_fam, config.id_hash_funcs, identity_features)
    
    if output_path is not None:
        with open(output_path, "wb") as pklfile:
            pickle.dump(img_nums, pklfile)
            pickle.dump(signals, pklfile)
            pickle.dump(concat_processed_signal, pklfile)
            pickle.dump(identity_features, pklfile)
            pickle.dump(dynamic_feat_hash, pklfile)
            pickle.dump(id_feat_hash, pklfile)
    
    #package the stuff 
    bin_seq_num = np.binary_repr(feature_seq_num, width = config.bin_seq_num_size)
    if feature_seq_num % 2 == 0: #use the correct half of the ID hash based on the sequence number. digest_process will ensure that the identity feature repeats every two times
        id_feat_hash_half = id_feat_hash[:config.identity_hash_k//2]
    else:
        id_feat_hash_half = id_feat_hash[config.identity_hash_k//2:]
    payload_p1 = bin_seq_num + id_feat_hash_half + dynamic_feat_hash

    # add a checksum
    payload_bytes = bitstring_to_bytes(payload_p1)
    checksum_gen = crc8.crc8()
    checksum_gen.update(payload_bytes)
    checksum = checksum_gen.digest()
    checksum_bits = bytes_to_bitstring(checksum)
    payload = payload_p1 + checksum_bits
    return payload, proc_signals, concat_processed_signal

    
class IdentityExtractor(object):
    def __init__(self):
        sys.stdout = open(os.devnull, "w")
        self.extractor = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.extractor.prepare(ctx_id=0, det_size=(640, 640))         
        sys.stdout = sys.__stdout__

    def extract(self, frame):
        faces = self.extractor.get(frame)
        if len(faces) == 0:
            return None
        e = faces[0]['embedding']
        normed_e = e / np.linalg.norm(e)
        return normed_e
        
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
        self.id_extractor = IdentityExtractor()
        self.video_path = video_path
    
    def get_id_features_hash(self, start_frame):
        cap = cv2.VideoCapture(self.video_path)
        frame_num = 0
        identity_features = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if not (frame_num >= start_frame):
                frame_num += 1
                continue
            if frame_num == start_frame:
                identity_features = self.id_extractor.extract(frame)
                break

        id_fam = config.id_fam
        id_hash_funcs = config.id_hash_funcs

        if identity_features is None:
            id_feat_hash = None
        else:
            id_feat_hash = hash_point(id_fam, id_hash_funcs, identity_features)
        return id_feat_hash, identity_features

    
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
    
