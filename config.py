import argparse
from main import main

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    # Dataset loading parameters
    parser.add_argument('-tr', '--train', nargs=3, metavar=('DATA_PATH', 'LABEL_PATH', 'PERCENT'), 
                        type=str, help="Path to training images, labels, and percentage (e.g., ./train_img ./train_lbl 80)")
    parser.add_argument('-va', '--validation', nargs=3, metavar=('DATA_PATH', 'LABEL_PATH', 'PERCENT'), 
                        type=str, help="Path to validation images, labels, and percentage (e.g., ./val_img ./val_lbl 10)")
    parser.add_argument('-te', '--test', nargs=3, metavar=('DATA_PATH', 'LABEL_PATH', 'PERCENT'), 
                        type=str, help="Path to test images, labels, and percentage (e.g., ./test_img ./test_lbl 10)")

    # Image processing parameters

    # Resize parser: user must specify the target size (width and height)
    parser.add_argument('--resize', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        help="Resize images to WIDTH x HEIGHT (e.g., 128 96).")

    # Feature extraction parser: user can specify one or more feature extraction techniques
    parser.add_argument('--features', nargs='+', choices=[
        'hog',        # Histogram of Oriented Gradients
        'lbp',        # Local Binary Patterns
        'sift',       # Scale-Invariant Feature Transform
        'surf',       # Speeded-Up Robust Features
        'orb',        # Oriented FAST and Rotated BRIEF
        'gabor',      # Gabor filters
        'haralick',   # Haralick texture features
        'color_hist', # Color histogram
        'glcm'        # Gray Level Co-occurrence Matrix
    ], help="Feature extraction techniques: hog, lbp, sift, surf, orb, gabor, haralick, color_hist, glcm")

    # Result parameters
    parser.add_argument('--result-type', choices=['all', 'accuracy', 'confusion_matrix', 'recall', 'f1_score'], nargs='+', default=['all'],
                        help="Result types: all, accuracy, confusion_matrix, recall, f1_score")
    args = parser.parse_args()

    #print("Parsing arguments...", args)
    main (args)
