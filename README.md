# Image Classifier Project

This project implements an image classification pipeline that allows for data loading, preprocessing, various feature extraction techniques, data augmentation, and classification using different machine learning models, including an ensemble approach. The pipeline is configurable via command-line arguments, offering flexibility in dataset handling, image processing, and model selection.


## Features

- **Flexible Dataset Handling**: Load training, validation, and testing datasets with specified percentages of data to use.
- **Image Preprocessing:**
Resizing: Images can be resized to custom dimensions.
- Feature Extraction: Supports a wide range of feature extraction techniques leveraging pyfeats, including:
    - `raw`: Raw pixel values
    - fos: First-order statistics
    - glcm: Gray Level Co-occurrence Matrix
    - glds: Gray Level Difference Statistics
    - ngtdm: Neighborhood Gray Tone Difference Matrix
    - sfm: Statistical Feature Matrix
    - lte: Local Ternary Patterns
    - fdta: Fractal Dimension Texture Analysis
    - glrlm: Gray Level Run Length Matrix
    - fps: Fourier Power Spectrum
    - shape: Shape features
    - glszm: Gray Level Size Zone Matrix
    - hos: Higher Order Statistics
    - lbp: Local Binary Patterns
    - grayscale_morphology: Grayscale morphological features
    - multilevel_binary_morphology: Multilevel binary morphological features
    - histogram: Histogram features
    - multiregion_histogram: Multi-region histogram features
    - correlogram: Color/texture correlogram
    - amfm: Adaptive Multi-scale Filter Moments
    - dwt: Discrete Wavelet Transform
    - swt: Stationary Wavelet Transform
    - wp: Wavelet Packet
    - gt: Gabor Transform
    - zernikes: Zernike moments
    - hu: Hu moments
    - tas: Texture Analysis Statistics
    - hog: Histogram of Oriented Gradients

- **Best Feature Selection**: Automatically identifies the best feature extraction technique for Random Forest, KNN, and SVM models based on F1-score.
- **Data Augmentation**: Includes techniques like mirroring, flipping, rotation, adding Gaussian noise, and changing brightness to balance datasets.
- **Model Training & Evaluation**:
    - **Supported Models**: Random Forest (rf), K-Nearest Neighbors (knn), Support Vector Machine (svm).
    - **Hyperparameter Tuning**: Uses Optuna for optimizing model hyperparameters to maximize F1-score.
    - **Ensemble Learning**: Implements a soft voting ensemble (ensemble) that combines predictions from trained Random Forest, KNN, and SVM models.
    - **Metrics**: Calculates and saves accuracy, precision, recall, F1-score, ROC AUC, and confusion matrices for each model.
- **HDF5 Integration**: Processes and saves image data into HDF5 files for efficient storage and retrieval.
- **Comprehensive Logging**: Detailed logs of program execution, arguments, and model training progress are saved.

## Installation

To set up the environment and install the necessary dependencies, you can use either `conda` with the `environment.yml` file or `pip` with the `requirements.txt` file.

**Using Conda**
```bash
conda env create -f environment.yml
conda activate image_classifier
```

**Using pip**
```bash
pip install -r requirements.txt
```

## Usage

The project can be run using the `config.py` script, which parses command-line arguments to control the image processing and classification pipeline.

**Command-line Arguments**
- **Dataset Parameters:**
    - `-t`, `--train`: Path to training images, labels, and an optional percentage (e.g., ``./train_img ./train_lbl 80``).
    - `-va`, `--validation`: Path to validation images, labels, and an optional percentage (e.g., ``./val_img ./val_lbl 10``).
    - ``-te``, ``--test``: Path to test images, labels, and an optional percentage (e.g., ``./test_img ./test_lbl 10``).
    - _Note:_ If any of ``train``, ``validation``, or ``test`` is provided, all three must be provided.

- **Image Processing Parameters:**
- ``--resize``: Resize images to WIDTH x HEIGHT (e.g., ``128 96``).

- ``-et``, ``--extract_technique``: Feature extraction techniques. Choose from: `raw`, `fos`, `glcm`, `glds`, `ngtdm`, `sfm`, `lte`, `fdta`, `glrlm`, `fps`, `shape`, `glszm`, `hos`, `lbp`, `grayscale_morphology`, `multilevel_binary_morphology`, `histogram`, `multiregion_histogram`, `correlogram`, `amfm`, `dwt`, `swt`, `wp`, `gt`, `zernikes`, `hu`, `tas`, `hog`, `best_feature`.
    - ``best_feature`` automatically selects the best features for each model. If ``best_feature`` is used, ``--model`` must not be specified as the script will train all models to find their best configurations.

- **Model Parameters:**
    - ``--model``: Choose one or more classification models: ``rf`` (Random Forest), ``knn`` (K-Nearest Neighbors), ``svm`` (Support Vector Machine), ``ensemble`` (Ensemble of RF, KNN, SVM), or ``all`` (trains all individual models and the ensemble).

### Examples
1. Run with specific extraction technique and model:

```bash
python config.py -tr ./data/train_images.png ./data/train_labels.txt 80 -va ./data/val_images.png ./data/val_labels.txt 10 -te ./data/test_images.png ./data/test_labels.txt 10 --resize 64 64 -et hog --model rf
```

This command will:

- Use 80% of training data, 10% of validation data, and 10% of test data.
- Resize images to 64x64 pixels.
- Extract Histogram of Oriented Gradients (HOG) features.
- Train and evaluate a Random Forest model.

2. Find the best feature for all models:

```bash
python config.py -tr ./data/train_images.png ./data/train_labels.txt -va ./data/val_images.png ./data/val_labels.txt -te ./data/test_images.png ./data/test_labels.txt --model all
```

This command will:

- Automatically iterate through all feature extraction techniques and resize options.
- Determine the best combination of technique and resize for Random Forest, KNN, and SVM models based on F1-score.
- Train the final models with their respective best configurations and evaluate an ensemble model.

3. Train all models with default settings (no specific feature extraction or resize):

```bash
python config.py -tr ./data/train_images.png ./data/train_labels.txt -va ./data/val_images.png ./data/val_labels.txt -te ./data/test_images.png ./data/test_labels.txt --model all
```

This assumes ``main.py`` handles default ``extraction_technique`` and ``resize`` if not provided, typically meaning raw pixel values without resizing. It will train and evaluate Random Forest, KNN, SVM, and the Ensemble model.

## Project Structure

```plaintext
.
├── config.py                 # Main configuration and argument parsing
├── main.py                   # Orchestrates the entire pipeline (data processing, model training)
├── requirements.txt          # Python dependencies for pip
├── environment.yml           # Conda environment file
├── .gitignore                # Files/directories to ignore by Git
├── dataset/
│   ├── __init__.py
│   └── data_processing.py    # Handles image loading, processing, augmentation, HDF5 generation
├── model/
│   ├── __init__.py
│   ├── Ensemble.py           # Implements the ensemble voting classifier
│   ├── KNN_Classifier.py     # K-Nearest Neighbors model
│   ├── Random_Forest.py      # Random Forest model
│   └── SVM_classifier.py     # Support Vector Machine model
├── utils/
│   ├── __init__.py
│   ├── augmentation.py       # Image augmentation techniques
│   ├── checks.py             # Argument validation functions
│   ├── image_utils.py        # Image feature extraction
│   ├── loogger.py            # Custom logging utility
│   └── process_train.py      # Helper functions for data processing and model training
├── Processed_images/         # Directory for generated HDF5 files (created during runtime)
│   └── image_processed.hdf5  # Processed image data (example)
├── LOG_TXT/
│   └── log.txt               # Detailed execution logs
└── Results/
    ├── Ensemble/
    │   ├── confusion_matrix.png
    │   ├── metrics.txt
    │   └── roc_curve.png
    ├── KNN/
    │   ├── confusion_matrix.png
    │   ├── metrics.txt
    │   └── roc_curve.png
    ├── RandomForest/
    │   ├── confusion_matrix.png
    │   ├── metrics.txt
    │   └── roc_curve.png
    └── SVM/
        ├── confusion_matrix.png
        ├── metrics.txt
        └── roc_curve.png
```

## Results
After running the classification pipeline, performance metrics (Accuracy, Precision, Recall, F1 Score, ROC AUC) and visualization (Confusion Matrix, ROC Curve) for each trained model (Random Forest, KNN, SVM, and Ensemble) are saved in their respective directories under the ``Results/`` folder.

For example, the metrics for the SVM model can be found in ``Results/SVM/metrics.txt``. Similarly, for Random Forest, results are in ``Results/RandomForest/metrics.txt``, for KNN in ``Results/KNN/metrics.txt``, and for the Ensemble model in ``Results/Ensemble/metrics.txt``.

## Logging
All program activities, including argument verification, dataset creation, and model training progress, are logged to ``LOG_TXT/log.txt``. This file provides a detailed chronological record of the execution.
