import os
from .loogger import Logger
import cv2
import glob
import sys
import shutil

def check_path_label(path, error_message):
    """
    Checks if the given path exists and is a file or directory containing label files.
    Accepts .txt or .csv files as valid label files.
    Returns the path to the .csv file if found, otherwise raises an Exception.

    Args:
        path (str): Path to check.
        error_message (str): Error message to raise if invalid.

    Raises:
        Exception: If the path does not exist or does not contain label files.

    Returns:
        str: Path to the .csv label file.
    """
    if not os.path.exists(path):
        raise Exception(error_message)
    label_extensions = {'.txt', '.csv'}
    if os.path.isdir(path):
        files = os.listdir(path)
        label_files = [f for f in files if os.path.splitext(f)[1].lower() in label_extensions]
        if not label_files:
            raise Exception(f"{error_message}: No label files found in directory.")
        # Prefer .csv file if available
        csv_files = [f for f in label_files if os.path.splitext(f)[1].lower() == '.csv']
        if csv_files:
            return os.path.join(path, csv_files[0])
        # If no .csv, return the first label file found
        return os.path.join(path, label_files[0])
    elif os.path.isfile(path):
        if os.path.splitext(path)[1].lower() not in label_extensions:
            raise Exception(f"{error_message}: File is not a supported label file.")
        return path
        

def check_path(path, error_message):
    """
    Checks if the given path exists and is a file or directory.
    If it's a directory, checks if it contains image files.
    Returns the image format (extension) if found.

    Args:
        path (str): Path to check.
        error_message (str): Error message to raise if invalid.

    Raises:
        Exception: If the path does not exist or does not contain images.

    Returns:
        str: The image format (e.g., '.jpg', '.png').
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    if not os.path.exists(path):
        raise Exception(error_message)
    if os.path.isdir(path):
        files = os.listdir(path)
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
        if not image_files:
            raise Exception(f"{error_message}: No image files found in directory.")
        # Return the extension of the first image file found
        return os.path.splitext(image_files[0])[1].lower()
    elif os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext not in image_extensions:
            raise Exception(f"{error_message}: File is not a supported image.")
        return ext

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
    Checks if the resize argument is a tuple/list of two positive numbers.
    
    Args:
        resize_tuple (tuple or list): Resize dimensions to check.
        error_message (str): Error message to raise if invalid.

    Raises:
        Exception: If the resize dimensions are invalid.
    Returns:
        tuple: (width, height) as floats
    """
    try:
        width = resize_tuple[0]
        height = float(resize_tuple[1])
    except (ValueError, TypeError):
        raise Exception(error_message)
    if width <= 0 or height <= 0:
        raise Exception(error_message)
    return resize_tuple 

def check_features(technique, error_message):
    """
    Checks if the features argument contains valid feature extraction techniques.

    Args:
        features (list or str): Feature extraction techniques to check.
        error_message (str): Error message to raise if invalid.

    Raises:
        Exception: If any feature extraction technique is invalid.
    """
    valid_techniques = {
        "raw", "fos", "glcm", "glds", "ngtdm", "sfm", "lte", "fdta",
        "glrlm", "fps", "shape", "glszm", "hos", "lbp", "grayscale_morphology",
        "multilevel_binary_morphology", "histogram", "multiregion_histogram",
        "correlogram", "amfm", "dwt", "swt", "wp", "gt", "zernikes", "hu", "tas", "hog","best_feature"
    }
    if isinstance(technique, str):
        technique = [technique]
    if not all(f in valid_techniques for f in technique):
        raise Exception(error_message)
    return technique 

def check_model(model, error_message):
    """
    Checks if the model argument contains valid model types.

    Args:
        model (list or str): Model types to check.
        error_message (str): Error message to raise if invalid.

    Raises:
        Exception: If any model type is invalid.
    Returns:
        list: Validated model types.
    """
    valid_models = {
        "svm", "random_forest", "knn", "ensemble"
    }
    if isinstance(model, str):
        model = [model]
    if not all(m in valid_models for m in model):
        raise Exception(error_message)
    return model

def verify_all_args(args):
    """
    Verifies all arguments for dataset, image processing, and result types.

    Args:
        args: Parsed arguments object.

    Raises:
        Exception: If any argument is invalid.
    """
    log = Logger(name="checks", level=10)

    log.info("Verifying all arguments...")
    
    # Ensure that if any of train, validation, or test is provided, all must be provided
    datasets = [args.train, args.validation, args.test]
    if any(ds is not None for ds in datasets):
        if not all(ds is not None for ds in datasets):
            raise Exception("If any of train, validation, or test is provided, all three must be provided.")

        # Ensure that at least one processing type is specified if datasets are provided
        if not (args.resize or args.extract_technique):
            log.error("You must specify at least one processing type: --resize or --extract_technique.")
            raise Exception("Processing type (--resize or/and --extract_technique) is required when train, validation, and test are provided.")
        try:
            check_path(args.train[0], "Invalid training path")
            train_label_path = check_path_label(args.train[1], "Invalid training labels path")
            if len(args.train) > 2:
                train_percent = check_porcentage(args.train[2], "Invalid training percentage")
            else:
                train_percent = 100.0
        except Exception as e:
            log.error(f"Error in training arguments: {e}")
            raise

        try:
            check_path(args.validation[0], "Invalid validation path")
            validation_label_path = check_path_label(args.validation[1], "Invalid validation labels path")
            if len(args.validation) > 2:
                validation_percent = check_porcentage(args.validation[2], "Invalid validation percentage")
            else:
                validation_percent = 100.0
        except Exception as e:
            log.error(f"Error in validation arguments: {e}")
            raise

        try:
            extention = check_path(args.test[0], "Invalid test path")
            test_label_path = check_path_label(args.test[1], "Invalid test labels path")
            if len(args.test) > 2:
                test_percent = check_porcentage(args.test[2], "Invalid test percentage")
            else:
                test_percent = 100.0
        except Exception as e:
            log.error(f"Error in test arguments: {e}")
            raise
        

        # Check if processed dataset already exists
        processed_images_path = os.path.join(os.getcwd(), 'Processed_images')
        if not os.path.isdir(processed_images_path):
            os.makedirs(processed_images_path)
            log.info(f"Created processed images directory at '{processed_images_path}'.")
        else:
            response = input(f"The folder '{processed_images_path}' already exists. Do you want to delete it? (y/n): ").strip().lower()
            if response == 'y':
                shutil.rmtree(processed_images_path)
                os.makedirs(processed_images_path)
                log.info(f"Deleted and recreated '{processed_images_path}'.")
            else:
                log.info("Exiting program as per user request.")
                sys.exit(0)

        # Image processing arguments verification
        if args.resize:
            try:
                # Check if resize dimensions are valid
                height_width = check_resize(args.resize, "Invalid resize dimensions")
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
                                resize_width, resize_height = height_width
                                if resize_width > orig_width or resize_height > orig_height:
                                    raise Exception(
                                        f"Resize dimensions must be lower than the original image size for {dataset_name} data"
                                    )
                
            except Exception as e:
                log.error(f"Error in resize dimensions: {e}")
                raise
        
        if args.extract_technique:
            try:
                extraction_technique = check_features(args.extract_technique, "Invalid extraction technique")
            except Exception as e:
                log.error(f"Error in extraction techniques: {e}")
                raise
    else:
        # If no dataset arguments are provided, set percentages to None
        train_percent = validation_percent = test_percent = None
    
    # Model verification
    if args.model:
        try:
            model = check_model(args.model, "Invalid result type")
        except Exception as e:
            log.error(f"Error in model: {e}")
            raise
    return {
        "train_label_path": train_label_path if 'train_label_path' in locals() else None,
        "extention": extention if 'extention' in locals() else None,
        "validation_label_path": validation_label_path if 'validation_label_path' in locals() else None,
        "test_label_path": test_label_path if 'test_label_path' in locals() else None,
        "train_percent": train_percent if 'train_percent' in locals() else None,
        "validation_percent": validation_percent if 'validation_percent' in locals() else None,
        "test_percent": test_percent if 'test_percent' in locals() else None,
        "resize_dims": height_width if 'height_width' in locals() else None,
        "extraction_technique": extraction_technique if 'extraction_technique' in locals() else None,
        "result_type": model if 'result_type' in locals() else None
    }

