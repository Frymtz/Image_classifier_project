from dataset import ImageDatasetGenerator
from model import *
from utils import Logger
from utils import checks as ch
import os

def main(args):
    log = Logger(name="main", level=10)
    log.info("Stating the program...")
#---------------------------------------------------------------------------------------------#
#Check parsers
    # Perform all argument verifications
    try:
        arg_results = ch.verify_all_args(args)
        train_label_path = arg_results.get("train_label_path")
        extention = arg_results.get("extention")
        validation_label_path = arg_results.get("validation_label_path")
        test_label_path = arg_results.get("test_label_path")
        train_percent = arg_results.get("train_percent")
        validation_percent = arg_results.get("validation_percent")
        test_percent = arg_results.get("test_percent")
        height_width = arg_results.get("resize_dims")
        extraction_technique = arg_results.get("extraction_technique")
        result_type = arg_results.get("result_type")
    except Exception as e:
        log = Logger(name="main.checks", level=10)
        log.error(f"Argument verification failed: {e}")
        raise
    log.info("All arguments verified successfully.")

    #print(f"Train: {train_percent}%, Validation: {validation_percent}%, Test: {test_percent}%")
#---------------------------------------------------------------------------------------------#
#Dataset + Processing
    log.info("Creating dataset...")
    try:
        generator = ImageDatasetGenerator(
            train_data_path=args.train[0],
            extension=extention,
            train_label_path=train_label_path,
            train_percent=train_percent,
            validation_data_path=args.validation[0],
            validation_label_path=validation_label_path,
            validation_percent=validation_percent,
            test_data_path=args.test[0],
            test_label_path= test_label_path,
            test_percent=test_percent,
            height_width=height_width,
            extraction_technique=extraction_technique
        )
        log.info("Dataset created successfully.")
    except Exception as e:
        log.error(f"Failed to create dataset generator: {e}")
        raise

#---------------------------------------------------------------------------------------------#
    # Generate HDF5 file
    try:
        output_path = os.path.join(os.getcwd(), "Processed_images/image_processed.hdf5")
        generator.generate_hdf5(output_path=output_path)
        log.info(f"HDF5 file generated at {output_path}")
    except Exception as e:
        log.error(f"Failed to generate HDF5 file: {e}")
        raise

#---------------------------------------------------------------------------------------------#
#Classification Model


if __name__ == "__main__":
    main()