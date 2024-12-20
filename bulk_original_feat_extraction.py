"""
Extract original dynamic features from a video
"""

import sys

sys.path.append("/home/hadleigh/deepfake_detection/system/e2e/common")
from digest_extraction import VideoDigestExtractor 

vd = VideoDigestExtractor("/home/hadleigh/verilight_attacks/data/test_video1.mp4")
_, _, dynamic_vec = vd.extract_from_video_slice(0, 100, 1, resample_signal=False, skip_hash=True)