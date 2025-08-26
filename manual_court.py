# %%
import sys
import math
import cv2 as cv
import numpy as np
import os

# %%
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
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    frame_num = fps*60 # 1 minute after recording starts


    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)

    cap.set(cv.CAP_PROP_POS_FRAMES, frame_num - 1)


    ret, src = cap.read()
    if not ret or src is None:
        cap.release()
        raise RuntimeError("Could not read first frame for playable area detection.")


    # %%
    cv.imshow('window', src)
    cv.waitKey(0)



    # %%
    cv.destroyAllWindows()


    # %%

    gaussian = cv.GaussianBlur(src, (5, 5), 0)
    cv.imshow("edge detection", gaussian)

    cv.waitKey(0)

    cv.destroyAllWindows()


    canny = cv.Canny(gaussian, 50, 200, None, 3)
    cv.imshow("edge detection", canny)

    cv.waitKey(0)

    cv.destroyAllWindows()

    height, width = canny.shape
    mask = np.zeros_like(canny)

    top_cutoff = int(height * 0.20)
    # The horizontal distance for the 45-degree corner cut
    corner_cut_width = int(width * 0.20) 
    # The vertical distance is the same as the horizontal for a 45-degree cut
    corner_cut_height = int(0.8 * corner_cut_width)

    # Define the points of the polygon we want to KEEP
    roi_vertices = np.array([
        (0, height),                                  # Bottom-left
        (0, top_cutoff + corner_cut_height),          # Left edge, below the corner cut
        (corner_cut_width, top_cutoff),               # Top edge, after the left corner cut
        (width - corner_cut_width, top_cutoff),       # Top edge, before the right corner cut
        (width, top_cutoff + corner_cut_height),      # Right edge, below the corner cut
        (width, height)                               # Bottom-right
    ], dtype=np.int32)

    # 5. Fill the polygon area on the mask with white (255)
    cv.fillPoly(mask, [roi_vertices], 255)

    # 6. Apply the mask to the Canny image using a bitwise AND operation
    # This keeps only the edges that are within the white area of the mask.
    masked_canny = cv.bitwise_and(canny, canny, mask=mask)
    cv.imshow("edge detection", masked_canny)

    cv.waitKey(0)

    cv.destroyAllWindows()






    cdstP = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)



    linesP = cv.HoughLinesP(masked_canny, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=30)





    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

    cv.imshow("Source", src)

    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv.waitKey(0)

    cv.destroyAllWindows()


    # %%


