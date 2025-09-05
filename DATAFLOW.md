# FILE TO DOCUMENT HOW DATA IS TRACKED THROUGH PIPELINE


- extract_all.py: saves pose data in npz format. Saves bounding boxes, keypoints, and keypoint confidence values to numpy arrays for EACH FRAME. If a frame is not selected for annotation, SAVES EMPTY FRAME DATA TO NPZ FILE. OUTPUT NPZ FILE HAS VIDEO FPS * LEN VIDEO SECONDS # ENTRIES

- filter_pose_data.py: iterates through each entry of the extract pose npz files, then filters out bbs that have a centroid outside of the court mask (calls court detector object once per file). Therefore, OUTPUT NPZ FILE HAS VIDEO FPS * LEN VIDEO SECONDS # ENTRIES. 

- data_processor.py: not sure



we already have existing annotation csv files. What we need is somewhere in our pipeline to create the target arrays. 



### PROBLEM: NO WAY TO TRACK WHICH FRAMES WERE SELECTED FOR INFERENCE AND WERE EMPTY VS FRAMES THAT NEVER HAD INFERENCE PERFORMED ON THEM. 

### SOLUTION: Add this to first inference file - load csv, and for each inference selected frame, if its within any of the [start_time,end_time intervals] (inclusive) then we give it target 1, else give it target 0. If the frame is not selected for inference, we give target -100 (negative and large, so clearly not the relevant frames)



### Cluttered pipeline: we have several saved npz files - original YOLO inference, filtered, then we data process. 

I think we should refactor to have 
1. original inference with yolo -> save to npz file (expensive, still debuggable s.t. we can see if yolo inference is bad or our data processing)
2. data preprocessing - the court mask filtering, combined with the data_processor methods for merging BBs and assigning players. This should save to a npz file, with an entry for every frame in the original video, the target (same as before, 0 or 1 or -100:marker for not annotated frame), and the near/far players labelled/annotated correctly. This decomposition is so that we can take the output preprocessed data npz file and draw it to the video, and see whether the data looks correct. 
3. feature engineering + data saving: This will ultimately create the indivual feature vectors for annotated frames (status>=0). It will take each annotated frame from the data preprocessing npz output file, then create all engineered features as already implemented. Its output will be an array of shape (n_annotated_frames, feature_vector size) as well as the target array of shape (n_annotated_frames,) for each video. It will save these to an aptly named directory. 

After this, our data has been fully processed, saved, and ready to prepare for training. 
4. we need a class to take the saved 