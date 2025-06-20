import os
from utils import Logger
from utils import checks as ch
from dataset import ImageDatasetGenerator
from model import  RandomForestModel, HardVotingEnsemble, KNNModel, SVMModel 


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

def process(generator, output_path, create_hdf5):
    X_train, X_val, X_test, y_train, y_val, y_test = generator.generate_hdf5(
        output_path=output_path, create_hdf5=create_hdf5
    )

    X_train = flatten_features(X_train)
    X_val = flatten_features(X_val)
    X_test = flatten_features(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_with_data(X_train, y_train, X_val, y_val, X_test, y_test, model, fit_trials=50):
    model_map = {
        "rf": RandomForestModel,
        "knn": KNNModel,
        "svm": SVMModel,
    }

    model_key = model.lower()
    model_instance = model_map[model_key]() if callable(model_map[model_key]) else model_map[model_key]()
    f1 = model_instance.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=fit_trials)

    return f1, model_instance

    
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

            if model[0].lower() == "rf":
                log.info("Training Random Forest model...")
                rf_model = RandomForestModel()
                rf_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                # metrics = rf_model.score(X_test, y_test)
                # log.info(f"Random Forest Metrics - F1-score: {metrics.get('f1')}, Accuracy: {metrics.get('accuracy')}, Precision: {metrics.get('precision')}, Recall: {metrics.get('recall')}")

            elif model[0].lower() == "knn":   
                log.info("Training KNN model...")
                knn_model = KNNModel()
                knn_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                # metrics = knn_model.score(X_test, y_test)
                # log.info(f"KNN Metrics - F1-score: {metrics.get('f1')}, Accuracy: {metrics.get('accuracy')}, Precision: {metrics.get('precision')}, Recall: {metrics.get('recall')}")
  
            elif model[0].lower() == "svm":
                log.info("Training SVM model...")
                svm_model = SVMModel()
                svm_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
            #     # metrics = svm_model.score(X_test, y_test)
            #     # log.info(f"SVM F1-score: {metrics['f1']}, Accuracy: {metrics.get('accuracy')}, Precision: {metrics.get('precision')}, Recall: {metrics.get('recall')}")
            else:
                rf_model = RandomForestModel()
                rf_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                knn_model = KNNModel()
                knn_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                svm_model = SVMModel()
                svm_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                ensemble = HardVotingEnsemble([rf_model, knn_model, svm_model])
            #     # ensemble_f1 = ensemble.score(X_test, y_test)
            #     # log.info(f"Ensemble Metrics - F1-score: {ensemble_f1.get('f1')}, Accuracy: {ensemble_f1.get('accuracy')}, Precision: {ensemble_f1.get('precision')}, Recall: {ensemble_f1.get('recall')}")

        except Exception as e:
            log.error(f"Failed to load data or train model: {e}")
            raise

    else:
        extraction_options = [
            ["raw"], ["fos"], ["glcm"], ["glds"], ["ngtdm"], ["sfm"], ["lte"], ["fdta"],
            ["glrlm"], ["fps"], ["shape"], ["glszm"], ["hos"], ["lbp"], ["grayscale_morphology"],
            ["multilevel_binary_morphology"], ["histogram"], ["multiregion_histogram"],
            ["correlogram"], ["amfm"], ["dwt"], ["swt"], ["wp"], ["gt"], ["zernikes"], ["hu"], ["tas"], ["hog"]
        ]

        extraction_options = [
            ["raw"], ["fos"] 
        ]

        # resize_options = [height_width, None]
        resize_options = [height_width]
        best_f1_rf = -1
        best_f1_knn = -1
        best_f1_svm = -1
        

        best_config = dict()

        for technique in extraction_options:
            for resize in resize_options:
                log.info(f"Processing extraction technique: {technique} with resize: {resize}")
                try:
                    generator = create_generator(
                        args, extention, train_label_path, train_percent,
                        validation_label_path, validation_percent,
                        test_label_path, test_percent, resize, technique
                    )
                    output_path = None
                    (
                        X_train, y_train,
                        X_val, y_val,
                        X_test, y_test 
                    ) = process(generator, output_path, False)

                    f1_rf, rf_model = train_with_data(X_train, y_train, X_val, y_val, X_test, y_test, model="rf", fit_trials=5)  
                    f1_knn, knn_model = train_with_data(X_train, y_train, X_val, y_val, X_test, y_test, model="knn", fit_trials=5)
                    f1_svm, svm_model = train_with_data(X_train, y_train, X_val, y_val, X_test, y_test, model="svm", fit_trials=5)
                    log.info(f"F1-score for {technique} | Resize: {resize} | Model: RANDOM FOREST: {f1_rf}")
                    log.info(f"F1-score for {technique} | Resize: {resize} | Model: KNN: {f1_knn}")
                    log.info(f"F1-score for {technique} | Resize: {resize} | Model: SVM: {f1_svm}")     
                    
                    if f1_rf > best_f1_rf:
                        best_f1_rf = f1_rf
                        best_config['rf'] = {'technique': technique, 'resize': resize}
                    if f1_knn > best_f1_knn:
                        best_f1_knn = f1_knn
                        best_config['knn'] = {'technique': technique, 'resize': resize}
                    if f1_svm > best_f1_svm:
                        best_f1_svm = f1_svm
                        best_config['svm'] = {'technique': technique, 'resize': resize}
               
                except Exception as e:
                    log.error(f"Error with extraction {technique} and resize {resize}: {e}")

        log.info(f"Random Forest - Technique: {best_config['rf']['technique']} | Resize: {best_config['rf']['resize']} | F1-score: {best_f1_rf}")
        log.info(f"KNN - Technique: {best_config['knn']['technique']} | Resize: {best_config['knn']['resize']} | F1-score: {best_f1_knn}")
        log.info(f"SVM - Technique: {best_config['svm']['technique']} | Resize: {best_config['svm']['resize']} | F1-score: {best_f1_svm}")
        exit(-1)
        # Final training with best config

        # Train the Random Forest model with the best configuration
        log.info("Training Random Forest model with best configuration...")
        generator = create_generator(
            args, extention, train_label_path, train_percent,
            validation_label_path, validation_percent,
            test_label_path, test_percent, best_config['rf']['resize'], best_config['rf']['technique']
        )
        output_path = None

        _, rf_model, X_test, y_test = train_with_data(generator, output_path, False, model="rf", fit_trials=50, return_test=True)
        
        knn_model = None
        # svm_model = None

        # Train KNN model with best config
        log.info("Training KNN model with best configuration...")
        generator_knn = create_generator(
            args, extention, train_label_path, train_percent,
            validation_label_path, validation_percent,
            test_label_path, test_percent, best_config['knn']['resize'], best_config['knn']['technique']
        )
        _, knn_model, _, _ = train_with_data(generator_knn, output_path, False, model="knn", fit_trials=50, return_test=True)

        # Train SVM model with best config
        # generator for SVM
        log.info("Training SVM model with best configuration...")
        generator_svm = create_generator(
            args, extention, train_label_path, train_percent,
            validation_label_path, validation_percent,
            test_label_path, test_percent, best_config['svm']['resize'], best_config['svm']['technique']
        )
        _, svm_model, _, _ = train_with_data(generator_svm, output_path, False, model="svm", fit_trials=50, return_test=True)


        log.info("Final Random Forest, KNN and SVM model trained with best configuration.")
        # UNCOMMENT ONLY IF YOU ALREADY HAVE THE BEST MODEL AND PRE-PROCESSED DATA
        # ensemble = HardVotingEnsemble([rf_model, knn_model, svm_model])
        # ensemble_f1 = ensemble.score(X_test, y_test)
        # log.info(f"Ensemble Metrics - F1-score: {ensemble_f1.get('f1')}, Accuracy: {ensemble_f1.get('accuracy')}, Precision: {ensemble_f1.get('precision')}, Recall: {ensemble_f1.get('recall')}")


if __name__ == "__main__":
    main()