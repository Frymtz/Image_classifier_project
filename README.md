# Image Classifier Project

This project implements an image classification pipeline that allows for data loading, preprocessing, several feature extraction techniques, data augmentation, and classification using different machine learning models, including an ensemble approach. The pipeline is configurable via command-line arguments, offering flexibility in dataset handling, image processing, and model selection.

## Features

- **Flexible Dataset Handling**: Load training, validation, and testing datasets with specified percentages of data to use.
- **Image Preprocessing:**

- **Resizing**: Images can be resized to custom dimensions to standardize input for feature extraction and model training. The minimum allowed dimension for both width and height is **24x24** pixels to ensure compatibility with all feature extraction techniques and avoid loss of critical image information.

This project leverages the [pyImageFeature (pyfeats)](https://github.com/giakoumoglou/pyfeats/tree/c0ad3abdaa60bb30853afce41afe5e7db6f52dda) repository to provide a comprehensive set of feature extraction techniques for image analysis. The `pyfeats` library is a well-established Python package that implements a wide variety of classical and advanced feature extraction methods, enabling robust texture, shape, and statistical analysis of images. By integrating `pyfeats`, this project supports reproducible and efficient extraction of features commonly used in medical imaging, pattern recognition, and computer vision research.

**Feature Extraction:**  
The pipeline supports a broad range of feature extraction techniques via `pyfeats`, including:

- `raw`: Raw pixel values
- `fos`: First-order statistics
- `glcm`: Gray Level Co-occurrence Matrix
- `glds`: Gray Level Difference Statistics
- `ngtdm`: Neighborhood Gray Tone Difference Matrix
- `sfm`: Statistical Feature Matrix
- `lte`: Local Ternary Patterns
- `fdta`: Fractal Dimension Texture Analysis
- `glrlm`: Gray Level Run Length Matrix
- `fps`: Fourier Power Spectrum
- `shape`: Shape features
- `glszm`: Gray Level Size Zone Matrix
- `hos`: Higher Order Statistics
- `lbp`: Local Binary Patterns
- `grayscale_morphology`: Grayscale morphological features
- `multilevel_binary_morphology`: Multilevel binary morphological features
- `histogram`: Histogram features
- `multiregion_histogram`: Multi-region histogram features
- `correlogram`: Color/texture correlogram
- `amfm`: Adaptive Multi-scale Filter Moments
- `dwt`: Discrete Wavelet Transform
- `swt`: Stationary Wavelet Transform
- `wp`: Wavelet Packet
- `gt`: Gabor Transform
- `zernikes`: Zernike moments
- `hu`: Hu moments
- `tas`: Texture Analysis Statistics
- `hog`: Histogram of Oriented Gradients

These techniques allow the extraction of rich and diverse descriptors from images, facilitating the training of robust machine learning models for classification tasks.

- **Best Feature Selection**: The pipeline includes an automated process to determine the most effective feature extraction technique for each classification model (Random Forest, KNN, and SVM) for a further ensemble approach. This is achieved by systematically evaluating all supported feature extraction methods and resize options, using the F1-score (macro) as the primary metric for comparison. The process selects the optimal combination of feature extraction technique and image size for each model, ensuring that the final ensemble leverages the strengths of each individual classifier. This approach maximizes overall classification performance and reduces the need for manual experimentation.
    **Important:** When using this feature, the `--model` flag must not be specified, as the script will train all supported models to determine their best configurations and then perform ensemble learning with them.

## Techniques Used for Balancing

The project uses two main approaches to address class imbalance in the dataset:

### 1. Data Augmentation

Data augmentation involves applying artificial transformations to images from the minority class to increase their quantity and diversity. The implemented techniques include:

- **Mirroring and Flipping**: Generates horizontally or vertically flipped versions of images.
- **Rotation**: Rotates images at different angles.
- **Adding Gaussian Noise**: Adds noise to images to simulate natural variations.
- **Brightness Adjustment**: Modifies image brightness to create new samples.

These transformations are applied to equalize the number of samples in each class, performing oversampling of the minority class and promoting a more balanced dataset for model training.

### 2. Class Weight

In addition to data augmentation, the project uses the `class_weight` parameter in classification algorithms (when supported, such as in Random Forest and SVM). This parameter adjusts the weight of each class during training, giving greater importance to minority classes. As a result, the model is penalized more heavily for errors in these classes, encouraging more balanced performance across all categories.

The combination of these techniques ensures that the model is trained robustly, reducing bias toward the majority class and improving generalization across all classes in the dataset.

## Model Training & Evaluation

- **Supported Models**:  
    The pipeline supports three widely used machine learning classifiers for image classification:
    - **Random Forest (rf)**: An ensemble of decision trees that aggregates predictions to improve accuracy and control overfitting.
    - **K-Nearest Neighbors (knn)**: A non-parametric method that classifies images based on the majority label among the k closest samples in the feature space.
    - **Support Vector Machine (svm)**: A robust classifier that finds the optimal hyperplane to separate classes in the feature space, effective for both linear and non-linear problems.

- **Hyperparameter Tuning with Optuna**:  
    To achieve optimal model performance, the pipeline integrates [Optuna](https://optuna.org/), an automatic hyperparameter optimization framework. For each supported model, Optuna systematically explores combinations of hyperparameters (such as the number of trees in Random Forest, the value of k in KNN, or kernel parameters in SVM) to maximize the macro F1-score on the validation set. This automated search ensures that each model is trained with the best possible configuration, reducing manual trial-and-error and improving classification results.
    - **Ensemble Learning**:  
        The project implements a hard voting ensemble classifier that combines the predictions of the individually trained Random Forest, KNN, and SVM models. In hard voting, each model casts a vote for a class label, and the class with the majority of votes is selected as the final prediction. This approach leverages the strengths of each model, often resulting in improved overall performance and robustness compared to any single classifier.

- **Evaluation Metrics & Visualization**:  
    For each trained model (including the ensemble), the following metrics are computed and saved:
    - **Accuracy**: Overall proportion of correctly classified samples.
    - **Precision**: Proportion of true positives among all predicted positives, calculated per class and averaged.
    - **Recall**: Proportion of true positives among all actual positives, calculated per class and averaged. _Recall is the primary metric for model selection and reporting._
    - **F1-score**: Harmonic mean of precision and recall, providing a balanced measure of model performance.
    - **ROC AUC**: Area under the Receiver Operating Characteristic curve, indicating the model's ability to distinguish between classes.
    - **Confusion Matrix**: A matrix visualization showing the counts of true positives, false positives, true negatives, and false negatives for each class.

    All metrics are saved as text files (e.g., `metrics.txt`), and visualizations such as confusion matrices and ROC curves are generated as images for each model in their respective directories under `Results/`. This comprehensive evaluation allows for detailed analysis and comparison of model performance.

- **HDF5 Integration**:  
    The pipeline processes all image data and stores it in HDF5 files, a format designed for fast, efficient storage and access of large datasets. By converting raw images and their extracted features into HDF5, the project minimizes memory usage and accelerates data loading during training and evaluation. This approach is especially beneficial for large-scale image datasets, as it allows for batch-wise reading and writing, supports compression, and ensures data consistency. The processed HDF5 files are automatically generated and saved in the `Processed_images/` directory, streamlining subsequent runs and enabling reproducible experiments.

- **Comprehensive Logging**:  
    The pipeline features robust logging of all key activities and events throughout execution. This includes validation of command-line arguments, dataset loading and preprocessing steps, feature extraction, model training progress, hyperparameter optimization, and evaluation results. All logs are saved to `LOG_TXT/log.txt`, providing a detailed, chronological record that facilitates debugging, reproducibility, and experiment tracking. The log file captures warnings, errors, and informational messages, ensuring transparency and traceability for every run.

## Inference Information (`--inference_info` flag)

The pipeline provides an optional `--inference_info` flag to facilitate model inference and result interpretation. When this flag is specified during execution, the script outputs detailed information about the inference process, including:

- The path to the HDF5 file containing the processed data.
- The path to the trained model to be used for inference.
- The dataset split to use for inference (e.g., `test`).

**Usage Example:**
```bash
python config.py --inference_info "./Processed_images/image_processed.hdf5;./Results/models/svm_model.joblib;test"
```

**How it works:**
- The script loads the specified dataset split (e.g., `test_data` and `test_label`) from the provided HDF5 file.
- It loads the trained model from the given path.
- The model performs predictions on the loaded data.
- Evaluation metrics (such as accuracy, precision, recall, F1-score, ROC AUC, and confusion matrix) are computed if ground truth labels are available.
- All inference results, including predicted class labels, confidence scores (if supported), and a summary of metrics, are saved to `Results/Inference/inference_report.txt`.

> **Important:**  
> To ensure correct inference results, the dataset stored in the HDF5 file must have undergone exactly the same preprocessing steps (such as resizing, normalization, feature extraction, etc.) as those used during model training.  
> Additionally, it is recommended to use the same version of `scikit-learn = 1.3.2` as was used for training, since version differences may cause incompatibilities when loading saved models.

## Installation

To set up the environment and install all required dependencies for this project, you have two options: using `conda` (recommended for reproducibility) or `pip`.

**Option 1: Using Conda**

The `environment.yml` file specifies all necessary packages and their versions. This ensures a consistent environment across different systems.

```bash
conda env create -f environment.yml
conda activate image_classifier
```

**Option 2: Using pip**

If you prefer to use `pip`, install the dependencies listed in `requirements.txt`. It is recommended to use a virtual environment (such as `venv` or `virtualenv`) to avoid conflicts with system packages.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**Note:**  
- The `pyfeats` library and other scientific packages may require additional system dependencies (such as build tools or libraries for image processing). If you encounter installation issues, refer to the documentation of the respective packages.
- After installation, you are ready to run the pipeline as described in the [Usage](#usage) section.

## Possible Issue with `pip install mahotas` and C++ Compiler

When installing the `mahotas` library using `pip`, you may encounter errors related to the absence of a C++ compiler. This is because `mahotas` contains C++ extensions that need to be compiled during installation.

**Common Error:**  
On Windows, you might see an error like:
```
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

**How to Resolve:**

- **Windows:**  
    1. Download and install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    2. After installation, restart your terminal and try `pip install mahotas` again.

- **Linux:**  
    Install the required build tools using your package manager. For example:
    ```bash
    sudo apt-get install build-essential
    ```

Alternatively, you can install a pre-built wheel (if available) or use `conda install mahotas` in a conda environment, which handles dependencies automatically.


## Usage

The project can be run using the `config.py` script, which parses command-line arguments to control the image processing and classification pipeline.
**Command-line Arguments**

- **Dataset Parameters:**
    - `-tr`, `--train`: Path to training images, labels, and an optional percentage of data to use (e.g., `./train_img ./train_lbl 80`). If the percentage is not specified, **100%** of the data is used by default.
    - `-va`, `--validation`: Path to validation images, labels, and an optional percentage (e.g., `./val_img ./val_lbl 10`). Defaults to **100%** if not specified.
    - `-te`, `--test`: Path to test images, labels, and an optional percentage (e.g., `./test_img ./test_lbl 10`). Defaults to **100%** if not specified.
    - _Note:_ If any of `train`, `validation`, or `test` is provided, all three must be specified to ensure consistent dataset splitting.

- **Image Processing Parameters:**
    - `--resize`: Resize images to the specified `WIDTH HEIGHT` (e.g., `128 96`). The minimum allowed size is 24x24 pixels.
    - `-et`, `--extract_technique`: Select the feature extraction technique. Options include: `raw`, `fos`, `glcm`, `glds`, `ngtdm`, `sfm`, `lte`, `fdta`, `glrlm`, `fps`, `shape`, `glszm`, `hos`, `lbp`, `grayscale_morphology`, `multilevel_binary_morphology`, `histogram`, `multiregion_histogram`, `correlogram`, `amfm`, `dwt`, `swt`, `wp`, `gt`, `zernikes`, `hu`, `tas`, `hog`, or `best_feature`.
        - `best_feature`: Automatically determines the optimal feature extraction technique and resize for each model. When using `best_feature`, do **not** specify `--model`; the script will train all supported models, select their best configurations, and build an ensemble.

- **Model Parameters:**
    - `--model`: Specify one or more classification models to train: `rf` (Random Forest), `knn` (K-Nearest Neighbors), `svm` (Support Vector Machine), `ensemble` (combines RF, KNN, and SVM), or `all` (trains all individual models and the ensemble).

**Note:**  
After training, each model is saved in `Results/models` for future use, so you do not need to rerun the entire pipeline to make predictions or reuse trained models.

### Examples

1. **Run with a specific feature extraction technique and model:**

```bash
python config.py -tr ./data/train_images.png ./data/train_labels.txt 80 -va ./data/val_images.png ./data/val_labels.txt 10 -te ./data/test_images.png ./data/test_labels.txt 10 --resize 64 64 -et hog --model rf
```

**What this does:**

- Uses 80% of the training data, 10% of the validation data, and 10% of the test data.
- Resizes all images to 64x64 pixels.
- Extracts Histogram of Oriented Gradients (HOG) features from each image.
- Trains and evaluates a Random Forest classifier using the extracted features.

This setup is useful when you want to experiment with a specific feature extraction method and model, allowing you to control both the preprocessing and the classification steps.

2. **Automatically select the best feature extraction technique and image size for all models:**

```bash
python config.py -tr ./data/train_images.png ./data/train_labels.txt -va ./data/val_images.png ./data/val_labels.txt -te ./data/test_images.png ./data/test_labels.txt -et best_feature
```

**What this does:**

- Systematically evaluates all supported feature extraction techniques and image resize options.
- Identifies the optimal combination of feature extraction method and image size for each model (Random Forest, KNN, and SVM) using the macro F1-score as the selection metric.
- Trains each model with its best configuration and then builds and evaluates an ensemble model that combines their predictions.

This approach automates the process of feature and parameter selection, ensuring that each classifier is trained with the most effective settings for your dataset, and maximizes overall classification performance.

> **Note:**  
> This feature should be used with caution when working with large datasets, as it requires significant computational power and may take considerable time for training and processing.

3. **Train all models using the same resize and feature extraction technique:**

```bash
python config.py -tr ./data/train_images.png ./data/train_labels.txt -va ./data/val_images.png ./data/val_labels.txt -te ./data/test_images.png ./data/test_labels.txt --model all
```

**What this does:**

- Trains all supported models (Random Forest, KNN, SVM, and the ensemble) using the current resize and feature extraction settings specified by the other command-line arguments (such as `--resize` and `-et`).
- Evaluates and saves the results for each model in their respective directories under `Results/`.

This option is useful when you want to compare the performance of all models using a consistent preprocessing and feature extraction configuration.

## Logging
All program activities, including argument verification, dataset creation, and model training progress, are logged to ``LOG_TXT/log.txt``. This file provides a detailed chronological record of the execution.

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

The results obtained from our Random Forest classifier demonstrate excellent performance across all key metrics, as evidenced by the quantitative measures and visualizations shown in Figure 1 (Confusion Matrix) and Figure 2 (ROC Curve). The model achieved an overall accuracy of 86.13%, with particularly strong performance in precision (88.97%) and recall (86.13%), indicating a well-balanced classifier that maintains high predictive power while effectively identifying positive cases.

The confusion matrix (Figure 1) reveals the model's detailed classification behavior:

- Exceptional specificity with 8,869 true negatives and only 33 false positives

- Strong sensitivity with 6,466 true positives, though showing 2,436 false negatives

- The relatively higher number of false negatives compared to false positives suggests the model is somewhat conservative in its positive predictions

The ROC curve (Figure 2) confirms the model's outstanding discriminative ability with an AUC-ROC of 0.972, approaching near-perfect classification performance. This near-ideal AUC score indicates the model can effectively separate the positive and negative classes across all classification thresholds.

Despite these strong results, we acknowledge the computational limitations that prevented us from employing our advanced 'best_feature' optimization technique. This feature, which automatically evaluates all 28 feature extraction methods to identify the optimal combination for each model, could potentially have improved these results further. The current implementation uses a single feature extraction approach rather than the optimized combination that 'best_feature' would have determined.

The F1-Score of 0.859 demonstrates good harmony between precision and recall, though we note that in medical applications like cancer detection represented by this dataset, we typically prioritize recall even higher to minimize missed positive cases. The current recall of 86.13% means about 14% of actual positive cases were missed - an area where we believe the 'best_feature' optimization could have helped reduce this percentage.

These results, while already strong, represent what we consider a baseline performance level. With additional computational resources to enable the full 'best_feature' optimization pipeline, we anticipate being able to:

- Further reduce the false negative rate

- Potentially improve the AUC score beyond 0.97

- Achieve better balance between precision and recall

- Identify more robust feature combinations specific to this dataset

The current implementation serves as a proof-of-concept for the classifier's capabilities, with the understanding that its full potential would be realized with access to greater computational resources for the complete feature optimization pipeline.


## Test Results

<p align="center">
  <img src="Results\RandomForest\confusion_matrix.png" alt="App Preview" width="400"/>
  <br>
  <em>Fig 1: Confusion Matrix obtained by RandoForest Model</em>
</p>

<p align="center">
  <img src="Results\RandomForest\roc_curve.png" alt="App Preview" width="400"/>
  <br>
  <em>Fig 2: ROC curve obtained by RandoForest Model</em>
</p>
