from dataset import ImageDatasetGenerator
from model import  RandomForestModel, SHVotingEnsemble, KNNModel, SVMModel

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

def train_with_data(X_train, y_train, X_val, y_val, model, fit_trials=50):
    model_map = {
        "rf": RandomForestModel,
        "knn": KNNModel,
        "svm": SVMModel,
    }

    model_key = model.lower()
    model_instance = model_map[model_key]() if callable(model_map[model_key]) else model_map[model_key]()
    f1 = model_instance.fit(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=fit_trials)

    return f1, model_instance

    
