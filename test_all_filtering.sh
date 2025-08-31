#!/bin/bash

# Test filtering pipeline on all pose data files in yolos_0.03conf_10.0fps_30s_to_90s directory
# Parameters: start_time=30, duration=60, target_fps=10, model_size=s, confidence=0.03

echo "🎯 Testing filtering pipeline on all pose data files"
echo "Parameters: start_time=30, duration=60, target_fps=10, model_size=s"
echo "================================================================"

# Define parameters
START_TIME=30
DURATION=60
TARGET_FPS=10
MODEL_SIZE="s"
CONFIDENCE="0.03"

# Array of video files that have corresponding pose data
declare -a videos=(
    "raw_videos/Verified UTR Match Play： Son Firewall 7.13 vs. Linh Tran (Unrated).mp4"
    "raw_videos/Verified UTR Match Play： Hung Ha 6.35 vs. Linh Tran (Unrated).mp4"
    "raw_videos/Verified UTR Match Play： Hung Ha 5.42 vs. Tuan Anh (Unrated).mp4"
    "raw_videos/Verified UTR Match Play： Hai Pham 4.63 vs. Tuan Anh (Unrated).mp4"
    "raw_videos/Unedited Points - Fall 2021 - Niki Stoiber.mp4"
    "raw_videos/Unedited Points - Fall 2021 - Erkin Tootoonchi Moghaddam.mp4"
    "raw_videos/Unedited Matchplay vs UTR 9.8.mp4"
    "raw_videos/Set (2) Play Uncut Derek vs. Alex.mp4"
    "raw_videos/Set (1) Play Uncut Derek vs. Alex.mp4"
    "raw_videos/Satoru Nakajima (11.75 UTR) Match Video (Unedited) vs. Dylan Chou (11.14 UTR) November 5, 2023.mp4"
    "raw_videos/Ryan Parkins - Unedited matchplay.mp4"
    "raw_videos/Otto Friedlein - unedited matchplay.mp4"
    "raw_videos/Match Play： Alex Sviatoslavsky Tennis Recruiting Video.mp4"
    "raw_videos/Holly Schlatter ｜ 2021 Australian Tennis Recruit  - Match Play Footage.mp4"
    "raw_videos/Brady Knackstedt (Blue Shirt⧸Black Shorts)(4.0 UTR) Unedited Match Play vs. opponent (5.54 UTR).mp4"
    "raw_videos/Anna Fijalkowska UNCUT MATCH PLAY (vs Felix Hein).mp4"
    "raw_videos/Aditi Narayan ｜ Matchplay.mp4"
    "raw_videos/9⧸5⧸15 Singles Uncut.mp4"
    "raw_videos/Monica Greene unedited tennis match play.mp4"
)

# Counter for tracking progress
total_videos=${#videos[@]}
current=0
successful=0
failed=0

echo "Found $total_videos videos to process"
echo ""

# Process each video
for video_path in "${videos[@]}"; do
    current=$((current + 1))
    video_name=$(basename "$video_path" .mp4)
    
    echo "🔄 Processing $current/$total_videos: $video_name"
    echo "   Video: $video_path"
    
    # Check if video file exists
    if [ ! -f "$video_path" ]; then
        echo "❌ Video file not found: $video_path"
        failed=$((failed + 1))
        continue
    fi
    
    # Run the filtering pipeline test
    echo "   Running: python test_filtering_pipeline.py $START_TIME $DURATION $TARGET_FPS $MODEL_SIZE \"$video_path\""
    
    if python test_filtering_pipeline.py $START_TIME $DURATION $TARGET_FPS $MODEL_SIZE "$video_path"; then
        echo "✅ Success: $video_name"
        successful=$((successful + 1))
    else
        echo "❌ Failed: $video_name"
        failed=$((failed + 1))
    fi
    
    echo ""
done

# Summary
echo "================================================================"
echo "🎯 FILTERING PIPELINE TEST SUMMARY"
echo "================================================================"
echo "Total videos processed: $total_videos"
echo "Successful: $successful"
echo "Failed: $failed"
echo "Success rate: $((successful * 100 / total_videos))%"

if [ $failed -eq 0 ]; then
    echo "🎉 All tests passed!"
    exit 0
else
    echo "❌ Some tests failed. Check the output above for details."
    exit 1
fi
