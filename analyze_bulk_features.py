import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import scipy.stats
sys.path.append("/home/hadleigh/deepfake_detection/system/e2e/common")
from digest_extraction import VideoDigestExtractor 

# extract original 
video_path = "/media/hadleigh/1TB/dynamic_window_level_clips/mod_clips_adjusted/10_20/dagan/charlie_25_45_close_1_1.mp4"
# get number of frames
cap = cv2.VideoCapture(video_path)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Number of frames: ", num_frames)
vd = VideoDigestExtractor(video_path)
_, _, dynamic_vec = vd.extract_from_video_slice(0, num_frames, 1, resample_signal=False, skip_hash=True)
print("Len of dynamic vec: ", len(dynamic_vec))

# load differentiable version
path_diff = "diff_dynamic_features/real_clips/charlie_0_45_close_0_diff_dynamic_vec.npy" 
vec1 = np.load(path_diff)

# compare
print("Len of diff dynamic vec: ", len(vec1))
plt.plot(vec1, label = "diff dynamic vec")
plt.plot(dynamic_vec, label = "dynamic vec")

# get pearson correlation of vecs
corr, _ = scipy.stats.pearsonr(vec1, dynamic_vec)

plt.legend()
plt.title(f"Pearson correlation: {corr}")
plt.savefig("charlie_25_45_close_0_1_diff_dynamic_vec.png")