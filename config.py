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
            'raw', 'fos', 'glcm', 'glds', 'ngtdm', 'sfm', 'lte', 'fdta',
            'glrlm', 'fps', 'shape', 'glszm', 'hos', 'lbp', 'grayscale_morphology',
            'multilevel_binary_morphology', 'histogram', 'multiregion_histogram',
            'correlogram', 'amfm', 'dwt', 'swt', 'wp', 'gt', 'zernikes', 'hu', 'tas', 'hog', 'best_feature'
        ],
        help=(
            "Feature extraction techniques:\n"
            "raw: Raw pixel values; "
            "fos: First-order statistics; "
            "glcm: Gray Level Co-occurrence Matrix; "
            "glds: Gray Level Difference Statistics; "
            "ngtdm: Neighborhood Gray Tone Difference Matrix; "
            "sfm: Statistical Feature Matrix; "
            "lte: Local Ternary Patterns; "
            "fdta: Fractal Dimension Texture Analysis; "
            "glrlm: Gray Level Run Length Matrix; "
            "fps: Fourier Power Spectrum; "
            "shape: Shape features; "
            "glszm: Gray Level Size Zone Matrix; "
            "hos: Higher Order Statistics; "
            "lbp: Local Binary Patterns; "
            "grayscale_morphology: Grayscale morphological features; "
            "multilevel_binary_morphology: Multilevel binary morphological features; "
            "histogram: Histogram features; "
            "multiregion_histogram: Multi-region histogram features; "
            "correlogram: Color/texture correlogram; "
            "amfm: Adaptive Multi-scale Filter Moments; "
            "dwt: Discrete Wavelet Transform; "
            "swt: Stationary Wavelet Transform; "
            "wp: Wavelet Packet; "
            "gt: Gabor Transform; "
            "zernikes: Zernike moments; "
            "hu: Hu moments; "
            "tas: Texture Analysis Statistics; "
            "hog: Histogram of Oriented Gradients; "
            "best_feature: Automatically selected best features."
        )
    )
    group_models = parser.add_argument_group('Model Parameters')
    group_models.add_argument('--model', choices=['rf', 'knn', 'svm', 'ensemble', 'all'], nargs='+',
                             help="Choose model(s): rf, knn, svm, ensemble")

    group_inference = parser.add_argument_group('Inference Parameters')
    group_inference.add_argument(
        '--evaluate_model',
        type=str,
        help=(
            "String with parameters separated by ';' in the order: "
            "hdf5_path;model_path;dataset\n"
            "Example: "
            "'./data/test.hdf5;./models/svm_model.h5;test'")
    )

    args = parser.parse_args()

    # Check if all arguments are None
    if (
        (not args.train or all(x is None for x in args.train)) and
        (not args.validation or all(x is None for x in args.validation)) and
        (not args.test or all(x is None for x in args.test)) and
        (not args.resize) and
        (not args.extract_technique) and
        (not args.model)
        and (not args.evaluate_model)
    ):
        log.error("All arguments are None. Please provide valid arguments.")
        raise Exception("All arguments are None. Please provide valid arguments.")

    main(args)
