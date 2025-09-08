def create_directory_structure(config):
    """Create the directory structure for the pipeline."""
    # Extract parameters from config
    start_time = config["start_time"]
    duration = config["duration"]
    fps = config["fps"]
    conf = config["conf"]
    model_size = config["model_size"]
    
    # Create directory name based on parameters - match pose_extractor naming convention
    dir_name = f"yolo{model_size}_{conf}conf_{fps}fps_{start_time}s_to_{duration}s"
    
    # Define directory paths
    unfiltered_dir = os.path.join("pose_data", "raw", dir_name)
    preprocessed_dir = os.path.join("pose_data", "preprocessed", dir_name)
    features_dir = os.path.join("pose_data", "features", dir_name)
    
    # Create directories if they don't exist
    os.makedirs(unfiltered_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    return unfiltered_dir, preprocessed_dir, features_dir