import argparse
from main import main
from utils import Logger

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description="Configuração do pipeline de processamento e classificação de imagens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    log = Logger(name="config", level=10)

    # Grupos de argumentos
    group_data = parser.add_argument_group('Parâmetros de Dataset')
    group_data.add_argument('-tr', '--train', nargs='+',
                           type=str, help="Caminho para imagens de treino, labels e porcentagem (ex: ./train_img ./train_lbl [80])")
    group_data.add_argument('-va', '--validation', nargs='+',
                           type=str, help="Caminho para imagens de validação, labels e porcentagem (ex: ./val_img ./val_lbl [10])")
    group_data.add_argument('-te', '--test', nargs='+',
                           type=str, help="Caminho para imagens de teste, labels e porcentagem (ex: ./test_img ./test_lbl [10])")

    group_img = parser.add_argument_group('Parâmetros de Processamento de Imagem')
    group_img.add_argument('--resize', type=int, nargs=2,
                          help="Redimensiona imagens para WIDTH x HEIGHT (ex: 128 96).")
    group_img.add_argument('-et','--extract_technique', nargs='+', choices=[
        'hog',        # Histogram of Oriented Gradients
        'lbp',        # Local Binary Patterns
        'sift',       # Scale-Invariant Feature Transform
        'surf',       # Speeded-Up Robust Features
        'orb',        # Oriented FAST and Rotated BRIEF
        'gabor',      # Gabor filters
        'haralick',   # Haralick texture features
        'color_hist', # Color histogram
        'glcm'        # Gray Level Co-occurrence Matrix
    ], help="Técnicas de extração de características: hog, lbp, sift, surf, orb, gabor, haralick, color_hist, glcm")

    group_result = parser.add_argument_group('Parâmetros de Resultados')
    group_result.add_argument('--result-type', choices=['all', 'accuracy', 'confusion_matrix', 'recall', 'f1_score'], nargs='+',
                             help="Tipos de resultado: all, accuracy, confusion_matrix, recall, f1_score")

    args = parser.parse_args()

    # Check if all arguments are None
    if (
        (not args.train or all(x is None for x in args.train)) and
        (not args.validation or all(x is None for x in args.validation)) and
        (not args.test or all(x is None for x in args.test)) and
        (not args.resize) and
        (not args.extract_feature) and
        (not args.result_type)
    ):
        log.error("All arguments are None. Please provide valid arguments.")
        raise Exception("All arguments are None. Please provide valid arguments.")

    main(args)
