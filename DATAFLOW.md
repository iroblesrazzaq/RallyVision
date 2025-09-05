# FILE TO DOCUMENT HOW DATA IS TRACKED THROUGH PIPELINE


- extract_all.py: saves pose data in npz format. Saves bounding boxes, keypoints, and keypoint confidence values to numpy arrays for EACH FRAME. If a frame is not selected for annotation, SAVES EMPTY FRAME DATA TO NPZ FILE. OUTPUT NPZ FILE HAS VIDEO FPS * LEN VIDEO SECONDS # ENTRIES

- filter_pose_data.py: iterates through each entry of the extract pose npz files, then filters out bbs that have a centroid outside of the court mask (calls court detector object once per file). Therefore, OUTPUT NPZ FILE HAS VIDEO FPS * LEN VIDEO SECONDS # ENTRIES. 

- data_processor.py: not sure