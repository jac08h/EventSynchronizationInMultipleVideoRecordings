# Configuration documentation

* demo: Clip pair for demo script.
    * clip_0: First clip.
    * clip_1: Second clip.
* tests:
    * clip_loaders:
        * use: Clip loaders to use.
        * augmented:
            * clip_directory: Directory containing augmented clips.
            * annotations_filename: Annotations for augmented clips.
        * easy:
            * clip_directory: Directory containing all replay clips.
            * annotations_file: Ground truth annotations for easy clips.
        * hard:
            * clip_directory: Directory containing all replay clips.
            * annotations_file: Ground truth annotations for hard clips.
    * metric: Which metric to use for evaluation - correct_frames/differences.
    * correct_frames: Count ratio of frames that were synchronized correctly.
        * unit: Unit of allowed error rates: frame/second.
        * allowed_error_rates: Allowed error rates for frame to be considered synchronized.
    * differences: Calculate mean/median differences between actual prediction and ground truth.
        * normalize: Divide the differences by maximum possible difference.
    * save_elapsed_time: Save average elapsed time.
* augmentations:
    * max_size: Maximum image size: (height, width). Set to null for no resizing.
    * ignore_top: Top fraction of image to ignore when finding keypoints.
    * ignore_bottom: Bottom fraction of image to ignore when finding keypoints.
* results:
    * output_directory: Directory for saving results.
    * clip_ext: Output clip extension.
    * plot_ext: Output plot extension.
    * save_plots: Save plot of cost matrix with predicted and ground truth path.
    * colormap: Cost matrix colormap from matplotlib.
    * visualization: What visualization to use: null/plain/keypoints
* cost_matrices:
    * device: Device to use for calculation.
    * fps: FPS for sampling the videos.
    * cache_directory: Directory with cached cost matrices.
    * use_cache: Try to used cached cost matrix if it exists. If it does not, calculate it and save to cache directory.
    * histogram_method: Method for histogram calculation: paper/simplified. Paper method refers to the one described in
      VideoSnapping paper, Equation 1. Simplified method only counts number of matches between frames.
    * histogram_parameters:
        * s_decay: Decay rate for spatial weights.
        * d_decay: Decay rate for descriptor weights.
    * cost_matrix_parameters:
        * alpha: Scaling parameter to penalize high cost bins.
    * opencv:
        * use: Which descriptors from OpenCV to use.
        * default_keypoint_detector: Used to detect keypoints if the descriptors does not have its own detector.
    * third_party:
        * use: Which neural descriptors from image-matching-toolbox to use.
        * D2Net:
            * model_path: Path to a pretrained model.
            * use_relu: Add ReLU as a final model layer. (See https://github.com/mihaidusmanu/d2-net/issues/16)
            * scales: Scales for detection as a list, e.g. [0.5, 1, 2]. [1] to use a single scale.
            * preprocessing: What kind of preprocessing to apply: caffe/torch.
        * SuperPoint:
            * keypoint_threshold: Score threshold for keypoints.
            * nms_radius: Radius for non-maximum suppression to remove nearby points.
            * remove_borders: Discard keypoints near image borders.
            * max_keypoints: Number of best keypoints to keep.
        * R2D2:
            * model_path: Path to a pretrained model.
            * top_k: Number of best keypoints to keep.
            * reliability_thr: Reliability threshold.
            * repeatability_thr: Repeatibility threshold.
            * min_scale: Minimum scale.
            * max_scale: Maximum scale.
    * kornia:
        * Which descriptors form kornia to use.
        * use_affnet: Use AffNet to estimate affine shape of patches detected by SIFT.
        * max_sift_features: Number of best features to retain from SIFT detection. If null, retain all features.
* path_finding:
    * partial_overlap:
        * use: Allow partial overlap.
        * min_path_length: Minimum path length if partial overlap is allowed.
    * warping_constraints:
        * use: Use warping constraints on slope.
        * path_length_for_slope_estimation: How many steps to go back to fo find start node for slope calculation.
        * min_slope: Minimum allowed slope.
        * max_slope: Maximum allowed slope.
        * penalty: Penalty for nodes that would break the allowed slope boundaries.