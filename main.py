import os
import joblib
from utils import Logger
from utils import checks as ch
from utils import ModelEvaluator
from model import RandomForestModel, SHVotingEnsemble, KNNModel, SVMModel 
from utils import flatten_features, process, create_generator, train_with_data
import h5py


def main(args):
    log = Logger(name="main", level=10)
    log.info("Stating the program...")

    # Inference mode
    if hasattr(args, 'evaluate_model') and args.evaluate_model:
        try:
            hdf5_path, model_path, dataset = args.evaluate_model.split(';')
            log.info(f"Inference mode: Loading data from {hdf5_path}, model from {model_path}, dataset: {dataset}")

            data = dataset + "_data"
            label = dataset + "_label"

            # Load data
            with h5py.File(hdf5_path, 'r') as f:
                X_test = f[data][:]
                y_test = f[label][:]
            X_test = flatten_features(X_test)

            # Load model
            model = joblib.load(model_path)
            evaluator = ModelEvaluator(model)
            evaluator.evaluate(X_test, y_test)

            log.info("Evaluate model results saved to Results/Inference/inference_report.txt")
            return
        except Exception as e:
            log.error(f"Evaluate model failed: {e}")
            raise
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
        try:
            data = generator.load_hdf5(output_path)
            X_train = flatten_features(data['train_data'])
            y_train = data['train_label']
            X_val = flatten_features(data['validation_data'])
            y_val = data['validation_label']
            X_test = data['test_data']
            X_test = flatten_features(data['test_data'])
            y_test = data['test_label']
            log.info("Data loaded successfully.")

            os.makedirs("Results/Trained_models", exist_ok=True)

            if model[0].lower() == "rf" or model[0].lower() == "all":
                log.info("Training Random Forest model...")
                rf_model = RandomForestModel()
                rf_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                rf_model.score(X_test, y_test)
                joblib.dump(rf_model, "Results/Trained_models/rf_model.joblib")
                log.info("Random Forest model saved as rf_model.joblib.")
                log.info("Random Forest model evaluation completed successfully.")

            if model[0].lower() == "knn" or model[0].lower() == "all":   
                log.info("Training KNN model...")
                knn_model = KNNModel()
                knn_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                knn_model.score(X_test, y_test)
                joblib.dump(knn_model, "Results/Trained_models/knn_model.joblib")
                log.info("KNN model saved as knn_model.joblib.")
                log.info("KNN model evaluation completed successfully.")

            if model[0].lower() == "svm" or model[0].lower() == "all":
                log.info("Training SVM model...")
                svm_model = SVMModel()
                svm_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                svm_model.score(X_test, y_test)
                joblib.dump(svm_model, "Results/Trained_models/svm_model.joblib")
                log.info("SVM model saved as svm_model.joblib.")
                log.info("SVM model evaluation completed successfully.")

            if model[0].lower() == "ensemble" or model[0].lower() == "all":
                rf_model = RandomForestModel()
                rf_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                knn_model = KNNModel()
                knn_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                svm_model = SVMModel()
                svm_model.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=50)
                ensemble = SHVotingEnsemble([rf_model, knn_model, svm_model], mode="default")
                
                ensemble.fit(X_train, y_train)
                ensemble.score(X_test, y_test)
                joblib.dump(ensemble, "Results/Trained_models/ensemble_model.joblib")
                log.info("Ensemble model saved as ensemble_model.joblib.")
                log.info("Ensemble model evaluation completed successfully.")

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
        # extraction_options = [
        #     ["raw"]
        
        # ]

        resize_options = [height_width, None]
        # resize_options = [height_width]

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

                    f1_rf, rf_model = train_with_data(X_train, y_train, X_val, y_val, model="rf", fit_trials=20)  
                    f1_knn, knn_model = train_with_data(X_train, y_train, X_val, y_val, model="knn", fit_trials=20)
                    f1_svm, svm_model = train_with_data(X_train, y_train, X_val, y_val, model="svm", fit_trials=20)
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

        # Final training with best config
        # Train the Random Forest model with the best configuration
        log.info("Training Random Forest model with best configuration...")
        generator = create_generator(
            args, extention, train_label_path, train_percent,
            validation_label_path, validation_percent,
            test_label_path, test_percent, best_config['rf']['resize'], best_config['rf']['technique']
        )
        output_path = None

        (   X_train, y_train,
            X_val, y_val,
            X_test_rf, y_test 
        ) = process(generator, output_path, False)

        _, rf_model = train_with_data(X_train, y_train, X_val, y_val, model="rf", fit_trials=50)
        
        knn_model = None
        svm_model = None

        # Train KNN model with best config
        log.info("Training KNN model with best configuration...")
        generator_knn = create_generator(
            args, extention, train_label_path, train_percent,
            validation_label_path, validation_percent,
            test_label_path, test_percent, best_config['knn']['resize'], best_config['knn']['technique']
        )
        
        (   X_train, y_train,
            X_val, y_val,
            X_test_knn, y_test 
        ) = process(generator_knn, output_path, False)

        _, knn_model = train_with_data(X_train, y_train, X_val, y_val, model="knn", fit_trials=50)

        # Train SVM model with best config
        # generator for SVM
        log.info("Training SVM model with best configuration...")
        generator_svm = create_generator(
            args, extention, train_label_path, train_percent,
            validation_label_path, validation_percent,
            test_label_path, test_percent, best_config['svm']['resize'], best_config['svm']['technique']
        )
        (   X_train, y_train,
            X_val, y_val,
            X_test_svm, y_test 
        ) = process(generator_svm, output_path, False)
                
        _, svm_model = train_with_data(X_train, y_train, X_val, y_val, model="svm", fit_trials=50)

        log.info("Final Random Forest, KNN and SVM model trained with best configuration.")
        
        ensemble = SHVotingEnsemble([rf_model, knn_model, svm_model], mode="multi_inputs")        
        ensemble.fit(X_train, y_train) #All models are already trained, so we just fit the ensemble for ROC AUC evaluation.
        
        X_test = [X_test_rf, X_test_knn, X_test_svm]
        ensemble.score(X_test, y_test)
        joblib.dump(ensemble, "Results/Trained_models/ensemble_model.joblib")
        log.info("Ensemble model saved as ensemble_model.joblib.")
        log.info("Ensemble model evaluation completed successfully.")
    log.info("Program completed successfully. :)")


if __name__ == "__main__":
    main()