import argparse
from main import main
from utils import Logger

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description="Configuration for the image processing and classification pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    log = Logger(name="config", level=10)

    # Argument groups
    group_data = parser.add_argument_group('Dataset Parameters')
    group_data.add_argument('-tr', '--train', nargs='+',
                           type=str, help="Path to training images, labels, and percentage (e.g.: ./train_img ./train_lbl [80])")
    group_data.add_argument('-va', '--validation', nargs='+',
                           type=str, help="Path to validation images, labels, and percentage (e.g.: ./val_img ./val_lbl [10])")
    group_data.add_argument('-te', '--test', nargs='+',
                           type=str, help="Path to test images, labels, and percentage (e.g.: ./test_img ./test_lbl [10])")

    group_img = parser.add_argument_group('Image Processing Parameters')
    group_img.add_argument('--resize', type=int, nargs=2,
                          help="Resize images to WIDTH x HEIGHT (e.g.: 128 96).")
    group_img.add_argument(
        '-et', '--extract_technique',
        nargs='+',
        choices=[
            'hog', 'lbp', 'sift', 'surf', 'orb', 'gabor', 'haralick', 'color_hist',
            'glcm', 'fos', 'glds', 'ngtdm', 'sfm', 'lte', 'fdta', 'glrlm', 'fps',
            'shape', 'glszm', 'hos', 'grayscale_morphology', 'multilevel_binary_morphology',
            'histogram', 'multiregion_histogram', 'correlogram', 'amfm', 'dwt', 'swt',
            'wp', 'gt', 'zernikes', 'hu', 'tas', 'best_feature'
        ],
        help=(
            "Feature extraction techniques:\n"
            "hog (Histogram of Oriented Gradients), "
            "lbp (Local Binary Patterns), "
            "sift (Scale-Invariant Feature Transform), "
            "surf (Speeded-Up Robust Features), "
            "orb (Oriented FAST and Rotated BRIEF), "
            "gabor (Gabor filters), "
            "haralick (Haralick texture features), "
            "color_hist (Color histogram), "
            "glcm (Gray Level Co-occurrence Matrix), "
            "fos (First Order Statistics), "
            "glds (Gray Level Difference Statistics), "
            "ngtdm (Neighborhood Gray Tone Difference Matrix), "
            "sfm (Statistical Feature Matrix), "
            "lte (Law's Texture Energy Measures), "
            "fdta (Fractal Dimension Texture Analysis), "
            "glrlm (Gray Level Run Length Matrix), "
            "fps (Fourier Power Spectrum), "
            "shape (Shape Parameters), "
            "glszm (Gray Level Size Zone Matrix), "
            "hos (Higher Order Spectra), "
            "grayscale_morphology (Gray-scale Morphological Analysis), "
            "multilevel_binary_morphology (Multilevel Binary Morphological Analysis), "
            "histogram (Histogram), "
            "multiregion_histogram (Multi-region histogram), "
            "correlogram (Correlogram), "
            "amfm (Amplitude Modulation – Frequency Modulation), "
            "dwt (Discrete Wavelet Transform), "
            "swt (Stationary Wavelet Transform), "
            "wp (Wavelet Packets), "
            "gt (Gabor Transform), "
            "zernikes (Zernikes’ Moments), "
            "hu (Hu’s Moments), "
            "tas (Threshold Adjacency Matrix),"
            "best_feature (Best Feature Selection Automatically)"
        )
    )
    group_models = parser.add_argument_group('Model Parameters')
    group_models.add_argument('--model', choices=['random_florest', 'knn', 'svm', 'ensemble'], nargs='+',
                             help="Choose model(s): random_florest, knn, svm, ensemble")

    args = parser.parse_args()

    # Check if all arguments are None
    if (
        (not args.train or all(x is None for x in args.train)) and
        (not args.validation or all(x is None for x in args.validation)) and
        (not args.test or all(x is None for x in args.test)) and
        (not args.resize) and
        (not args.extract_feature) and
        (not args.model)
    ):
        log.error("All arguments are None. Please provide valid arguments.")
        raise Exception("All arguments are None. Please provide valid arguments.")

    main(args)
