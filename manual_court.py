
# %%
import sys
import math
import cv2
import numpy as np
import os
# %%
def main():
    # cell 1: get image 1 minute in - in prod have to make sure nothing is covering doubles lines at that point (players)

    vid_paths = ['raw_videos/Aditi Narayan ｜ Matchplay.mp4', 'raw_videos/Monica Greene unedited tennis match play.mp4', 
                'raw_videos/Anna Fijalkowska UNCUT MATCH PLAY (vs Felix Hein).mp4',
                'raw_videos/Otto Friedlein - unedited matchplay.mp4']
    vid_paths1 = [f"./raw_videos/{filename}" for filename in os.listdir('raw_videos') if 'mp4' in filename]
        # %%
    video_path = 'raw_videos/Monica Greene unedited tennis match play.mp4'
    video_path = vid_paths[0]
    for video_path in vid_paths1:
        print(video_path.split('/')[-1])
        # %%
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_num = fps*60 # 1 minute after recording starts

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)

        ret, src = cap.read()
        if not ret or src is None:
            cap.release()
            raise RuntimeError("Could not read first frame for playable area detection.")

        # %%
        cv2.imshow('window', src)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        # --- 1. PRE-PROCESSING FOR EDGE DETECTION ---
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # --- 2. GENERATE THE TWO INITIAL MASKS ---
        # a) Find all high-contrast edges with Canny
        canny_edges = cv2.Canny(blurred, 50, 150)

        # b) Find all white pixels using LAB color space (the "loose" mask)
        lab_image = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        lower_white = np.array([135, 105, 105])
        upper_white = np.array([255, 150, 150])
        white_mask = cv2.inRange(lab_image, lower_white, upper_white)

        # --- 3. CREATE THE "PROXIMITY MASK" FROM CANNY EDGES ---
        # Dilate the Canny edges to create a zone around them.
        # A larger kernel size increases the "distance" threshold.
        kernel = np.ones((3,3), np.uint8)
        dilated_edges_mask = cv2.dilate(canny_edges, kernel, iterations=1)

        # --- 4. FIND THE INTERSECTION TO GET THE REFINED MASK ---
        # Keep only the pixels that are in BOTH the white mask AND the dilated edge mask
        refined_lines_mask = cv2.bitwise_and(white_mask, dilated_edges_mask)

        # (Optional) Clean up the final mask by closing small gaps
        refined_lines_mask = cv2.morphologyEx(refined_lines_mask, cv2.MORPH_CLOSE, kernel)

        # --- 5. VISUALIZE THE RESULTS ---
        final_result = cv2.bitwise_and(src, src, mask=refined_lines_mask)

        cv2.imshow('Original "Loose" White Mask', white_mask)
        cv2.imshow('Dilated Canny Edges (Zone of Interest)', dilated_edges_mask)
        cv2.imshow('Refined Lines Mask (Final)', refined_lines_mask)


        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        height, width = src.shape[:2]

        # --- Create a single-channel (grayscale) mask ---
        # Masks must be single-channel arrays of type uint8.
        roi_mask = np.zeros((height, width), dtype=np.uint8)

        # Define the cutoff points for the top and corners
        top_cutoff = int(height * 0.20)
        corner_cut_width = int(width * 0.20)
        corner_cut_height = int(0.8 * corner_cut_width)

        # Define the vertices of the polygon you want to KEEP
        roi_vertices = np.array([
            (0, height),                                  # Bottom-left
            (0, top_cutoff + corner_cut_height),          # Left edge, below the corner cut
            (corner_cut_width, top_cutoff),               # Top edge, after the left corner cut
            (width - corner_cut_width, top_cutoff),       # Top edge, before the right corner cut
            (width, top_cutoff + corner_cut_height),      # Right edge, below the corner cut
            (width, height)                               # Bottom-right
        ], dtype=np.int32)

        # Fill the polygon area on the single-channel mask with white (255)
        cv2.fillPoly(roi_mask, [roi_vertices], 255)

        # Apply this ROI mask to your final color result
        masked_result = cv2.bitwise_and(refined_lines_mask, refined_lines_mask, mask=roi_mask)


        cv2.imshow("ROI Mask", roi_mask)
        cv2.imshow("masked result", masked_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

                
        # (Your existing code to get linesP...)
# (Your existing code to get linesP...)
        linesP = cv2.HoughLinesP(masked_result, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=40)

        merged_lines_image = src.copy()
        """
        if merged_lines:
            for line in merged_lines:
                x1, y1, x2, y2 = line
                # Draw merged lines in green
                cv2.line(merged_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)
        """
        # --- Draw the ORIGINAL, UNMERGED lines for comparison ---
        unmerged_lines_image = src.copy()
        if linesP is not None:
            for line in linesP:
                x1, y1, x2, y2 = line[0]
                # Draw unmerged lines in red
                cv2.line(unmerged_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

        # --- Display both images ---
        # Added a check in case linesP is None to prevent an error
        print(f"num original hough lines: {len(linesP) if linesP is not None else 0}")#\nnum merged lines: {len(merged_lines)}")
        
        cv2.imshow('Unmerged Lines (Original)', unmerged_lines_image)
        #cv2.imshow("Merged Lines (Cleaned)", merged_lines_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #break

if __name__ == "__main__":
    main()