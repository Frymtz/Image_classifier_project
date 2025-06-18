from cycler import K
from matplotlib.pylab import f
from dataset import ImageDatasetGenerator
from model import  RandomForestModel
from utils import Logger
from utils import checks as ch
import os
from tqdm import tqdm
from model import KNNModel, SVMModel, HardVotingEnsemble

def create_generator(args, extention, train_label_path, train_percent, validation_label_path, validation_percent,
                    test_label_path, test_percent, height_width, extraction_technique):
    
    return ImageDatasetGenerator(
        train_data_path=args.train[0],
        extension=extention,
        train_label_path=train_label_path,
        train_percent=train_percent,
        validation_data_path=args.validation[0],
        validation_label_path=validation_label_path,
        validation_percent=validation_percent,
        test_data_path=args.test[0],
        test_label_path=test_label_path,
        test_percent=test_percent,
        height_width=height_width,
        extraction_technique=extraction_technique
    )

def flatten_features(X):
    if X is None:
        return None
    if len(X.shape) > 2:
        return X.reshape((X.shape[0], -1))
    return X

def process_and_train(generator, output_path, create_hdf5, log, fit_trials=50, return_test=False):

    X_train, X_val, X_test, y_train, y_val, y_test = generator.generate_hdf5(output_path=output_path, create_hdf5=create_hdf5)    
    
    X_train = flatten_features(X_train)
    X_val = flatten_features(X_val)
    X_test = flatten_features(X_test)

    rf_model = RandomForestModel()
    f1 = rf_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=fit_trials)
    if return_test:
        return f1, rf_model, X_test, y_test
    return f1, rf_model

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
        model = arg_results.get("model")
    except Exception as e:
        log = Logger(name="main.checks", level=10)
        log.error(f"Argument verification failed: {e}")
        raise
    log.info("All arguments verified successfully.")
#---------------------------------------------------------------------------------------------#
#Dataset + Processing

    if extraction_technique is None:
        ext_tech = None
    else:
        ext_tech = extraction_technique[0].lower()

    if ext_tech != "best_feature":
        log.info("Creating dataset...")
        try:
            generator = create_generator(
                args, extention, train_label_path, train_percent,
                validation_label_path, validation_percent,
                test_label_path, test_percent, height_width, extraction_technique
            )
            log.info("Dataset created successfully.")
        except Exception as e:
            log.error(f"Failed to create dataset generator: {e}")
            raise
#---------------------------------------------------------------------------------------------#
    # Generate HDF5 file
        try:
            output_path = os.path.join(os.getcwd(), "Processed_images/image_processed.hdf5")
            generator.generate_hdf5(output_path=output_path, create_hdf5=True)
            log.info(f"HDF5 file generated at {output_path}")
        except Exception as e:
            log.error(f"Failed to generate HDF5 file: {e}")
            raise

#---------------------------------------------------------------------------------------------#
    # Classification Model
    if ext_tech != "best_feature":
        log.info("Starting Random Forest model training...")
        try:
            data = generator.load_hdf5(output_path)
            X_train = data['train_data']
            y_train = data['train_label']
            X_val = data['validation_data']
            y_val = data['validation_label']
            X_train = flatten_features(X_train)
            X_val = flatten_features(X_val)

            # X_test = data['test_data']
            # X_test = flatten_features(data['test_data'])
            #y_test = data['test_label']
            log.info("Data loaded successfully.")

            if model.lower() == "rf":
                log.info("Training Random Forest model...")
                rf_model = RandomForestModel()
                rf_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                metrics = rf_model.score(X_test, y_test)
                print(f"Random Forest F1-score: {metrics['f1']}")

            elif model.lower() == "knn":   
                log.info("Training KNN model...")
                knn_model = KNNModel()
                knn_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                metrics = knn_model.score(X_test, y_test)
                print(f"KNN F1-score: {metrics['f1']}")
            elif model.lower() == "svm":
                log.info("Training SVM model...")
                svm_model = SVMModel()
                svm_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                metrics = svm_model.score(X_test, y_test)
                print(f"SVM F1-score: {metrics['f1']}")
            else:
                rf_model = RandomForestModel()
                rf_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                knn_model = KNNModel()
                knn_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                svm_model = SVMModel()
                svm_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                ensemble = HardVotingEnsemble([rf_model, knn_model, svm_model])
                # ensemble_f1 = ensemble.score(X_test, y_test)
                # print(f"Ensemble F1-score: {ensemble_f1}")

        except Exception as e:
            log.error(f"Failed to load data or train model: {e}")
            raise

    else:
        # extraction_options = [
        #     ["raw"], ["glcm"] 
        # ]

        resize_options = [height_width, None]
        best_f1_rt = -1
        best_f1_svm = -1
        best_f1_knn = -1
        best_config = None

        for technique in extraction_options:
            for resize in resize_options:
                try:
                    generator = create_generator(
                        args, extention, train_label_path, train_percent,
                        validation_label_path, validation_percent,
                        test_label_path, test_percent, resize, technique
                    )
                    output_path = None
                    f1, _ = process_and_train(generator, output_path, False, log, fit_trials=20)
                    log.info(f"F1-score for {technique} | Resize: {resize}: {f1}")
                    if f1 > best_f1:
                        best_f1 = f1
                        best_config = (technique, resize)
                except Exception as e:
                    log.error(f"Error with extraction {technique} and resize {resize}: {e}")

        log.info(f"Best extraction technique: {best_config[0]} | Resize: {best_config[1]} | F1-score: {best_f1}")

        # Final training with best config
        generator = create_generator(
            args, extention, train_label_path, train_percent,
            validation_label_path, validation_percent,
            test_label_path, test_percent, best_config[1], best_config[0]
        )
        output_path = None

        _, rf_model, X_test, y_test = process_and_train(generator, output_path, False, log, fit_trials=50, return_test=True)


        
        
        # UNCOMMENT ONLY IF YOU ALREADY HAVE THE BEST MODEL AND PRE-PROCESSED DATA
        # ensemble = HardVotingEnsemble([rf_model, knn_model, svm_model])
        # ensemble_f1 = ensemble.score(X_test, y_test)
        # y_pred = rf_model.predict(X_test)
        # metrics = rf_model.score(X_test, y_test)

if __name__ == "__main__":
    main()