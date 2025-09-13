# file to run inference on an entire feature npz file, then apply postprocessing steps
# to output final start_time,end_time csv file
import numpy as np
"""
My test.py file is my current evaluation file. however, it just looks at sequences, not the whole video. 
I need to further establish my post processing pipeline. The final output of my pipeline should be 
a csv of start_time, end_times, which i can compare to the annotated targets. 
For now, we will use the same gaussian smoothing and hysteresis filtering that we're using in the test.py file. 

Your task is to write a new file that runs the inference on an entire video's sequence file
"""

# steps: 

# load model - best 300 sequence length model
model_path = 'checkpoints/seq_len300/best_model.pth'


# load whole feature npz file for a specific video
video_feature_path = 'pose_data/features/Monica Greene unedited tennis match play.npz'
feature_data = np.load(video_feature_path)

# create our ordered list of sequences with 50% overlap: must carefully track frame numbers
'''
can use numpy ops, but have to track start, end frame. ignore un-annotated frames. 

num_frames = 



'''


# perform inference on each individual sequence


# now create final output sequence by merging all sequences, averaging all overlapping frame predictions. 

# perform gaussian smoothing on that probability sequence

# perform hysteresis filtering on smoothed sequence

# use hysteresis for start/end times, write to csv
