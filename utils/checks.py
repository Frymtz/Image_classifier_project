import os
import re
from .loogger import Logger
import cv2
import glob

def check_path(path, error_message):
    """
    Checks if the given path exists and is a file or directory.
    If it's a directory, checks if it contains image files.

    Args:
        path (str): Path to check.
        error_message (str): Error message to raise if invalid.

    Raises:
        Exception: If the path does not exist or does not contain images.
    """
    if not os.path.exists(path):
        raise Exception(error_message)
    if os.path.isdir(path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        files = os.listdir(path)
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
        if not image_files:
            raise Exception(f"{error_message}: No image files found in directory.")
    elif os.path.isfile(path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        if os.path.splitext(path)[1].lower() not in image_extensions:
            raise Exception(f"{error_message}: File is not a supported image.")

def check_porcentage(value, error_message, dataset_path=None):
    """
    Checks if the given value is a valid percentage (0 < value <= 100).
    If the dataset contains only 1 image, or if value is None, sets the percentage to 100.

    Args:
        value (float): Percentage value to check.
        error_message (str): Error message to raise if invalid.
        dataset_path (str, optional): Path to the dataset to check image count.

    Raises:
        Exception: If the value is not a valid percentage.
    Returns:
        float: The validated (or adjusted) percentage value.
    """
    if value is None:
        return 100.0
    if dataset_path:
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
        if os.path.isdir(dataset_path):
            files = os.listdir(dataset_path)
            image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
            if len(image_files) == 1:
                return 100.0
        elif os.path.isfile(dataset_path):
            if os.path.splitext(dataset_path)[1].lower() in image_extensions:
                return 100.0
    try:
        val = float(value)
    except (TypeError, ValueError):
        raise Exception(error_message)
    if not (0 < val <= 100):
        raise Exception(error_message)
    return val

def check_resize(resize_tuple, error_message):
    """
    Checks if the resize argument is a tuple of two positive integers.

    Args:
        resize_tuple (tuple): Resize dimensions to check.
        error_message (str): Error message to raise if invalid.

    Raises:
        Exception: If the resize dimensions are invalid.
    """
    if (not isinstance(resize_tuple, (tuple, list)) or
        len(resize_tuple) != 2 or
        not all(isinstance(x, int) and x > 0 for x in resize_tuple)):
        raise Exception(error_message)

def check_features(features, error_message):
    """
    Checks if the features argument contains valid feature extraction techniques.

    Args:
        features (list or str): Feature extraction techniques to check.
        error_message (str): Error message to raise if invalid.

    Raises:
        Exception: If any feature extraction technique is invalid.
    """
    valid_features = {
        "hog",
        "lbp",
        "sift",
        "surf",
        "orb",
        "gabor",
        "haralick",
        "color_hist",
        "glcm"
    }
    if isinstance(features, str):
        features = [features]
    if not all(f in valid_features for f in features):
        raise Exception(error_message)

def check_result_types(result_type, error_message):
    """
    Checks if the result_type argument contains valid result types.

    Args:
        result_type (list or str): Result types to check.
        error_message (str): Error message to raise if invalid.

    Raises:
        Exception: If any result type is invalid.
    """
    valid_types = {"all", "accuracy", "confusion_matrix", "recall", "f1_score"}
    if isinstance(result_type, str):
        result_type = [result_type]
    if not all(rt in valid_types for rt in result_type):
        raise Exception(error_message)

def verify_all_args(args):
    """
    Verifies all arguments for dataset, image processing, and result types.

    Args:
        args: Parsed arguments object.

    Raises:
        Exception: If any argument is invalid.
    """
    log = Logger(name="main.checks", level=10)

    log.info("Verifying all arguments...")
    
    # Ensure that if any of train, validation, or test is provided, all must be provided
    datasets = [args.train, args.validation, args.test]
    if any(ds is not None for ds in datasets):
        if not all(ds is not None for ds in datasets):
            raise Exception("If any of train, validation, or test is provided, all three must be provided.")


    # Dataset arguments verification
    try:
        check_path(args.train[0], "Invalid training path")
        check_path(args.train[1], "Invalid training labels path")
        if len(args.train) > 2:
            train_percent = check_porcentage(args.train[2], "Invalid training percentage")
        else:
            train_percent = 100.0
    except Exception as e:
        log.error(f"Error in training arguments: {e}")
        raise

    try:
        check_path(args.validation[0], "Invalid validation path")
        check_path(args.validation[1], "Invalid validation labels path")
        if len(args.validation) > 2:
            validation_percent = check_porcentage(args.validation[2], "Invalid validation percentage")
        else:
            validation_percent = 100.0
    except Exception as e:
        log.error(f"Error in validation arguments: {e}")
        raise

    try:
        check_path(args.test[0], "Invalid test path")
        check_path(args.test[1], "Invalid test labels path")
        if len(args.test) > 2:
            test_percent = check_porcentage(args.test[2], "Invalid test percentage")
        else:
            test_percent = 100.0
    except Exception as e:
        log.error(f"Error in test arguments: {e}")
        raise

    # Image processing arguments verification
    if args.resize:
        try:
            # Check if resize dimensions are valid
            check_resize(args.resize, "Invalid resize dimensions")
            # Check if resize dimensions are lower than the original image
            # Check resize against original image size for train, validation, and test datasets
            for dataset_name, dataset in [("train", args.train), ("validation", args.validation), ("test", args.test)]:
                if dataset and dataset[0]:
                    image_paths = []
                    if os.path.isdir(dataset[0]):
                        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif')
                        for ext in image_extensions:
                            image_paths.extend(glob.glob(os.path.join(dataset[0], ext)))
                    elif os.path.isfile(dataset[0]):
                        image_paths = [dataset[0]]
                    if image_paths:
                        img = cv2.imread(image_paths[0])
                        if img is not None:
                            orig_height, orig_width = img.shape[:2]
                            resize_width, resize_height = args.resize
                            if resize_width > orig_width or resize_height > orig_height:
                                raise Exception(
                                    f"Resize dimensions must be lower than the original image size for {dataset_name} data"
                                )
        except Exception as e:
            log.error(f"Error in resize dimensions: {e}")
            raise

    if args.extract_feature:
        try:
            check_features(args.extract_feature, "Invalid feature extraction technique")
        except Exception as e:
            log.error(f"Error in feature extraction techniques: {e}")
            raise

    # Result types verification
    if args.result_type:
        try:
            check_result_types(args.result_type, "Invalid result type")
        except Exception as e:
            log.error(f"Error in result types: {e}")
            raise
    
    return train_percent, validation_percent, test_percent
    

