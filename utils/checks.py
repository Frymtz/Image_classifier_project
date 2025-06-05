def check_path_and_percentage(path, perc, name):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"O caminho '{path}' para {name} não existe.")
    try:
        perc = float(perc)
        if not (0 < perc <= 100):
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"A porcentagem '{perc}' para {name} deve ser um número entre 0 e 100.")
    return path, perc


def parse_path_perc(arg):
    try:
        path, perc = arg.split(',')
        return check_path_and_percentage(path.strip(), perc.strip(), path)
    except Exception:
        raise argparse.ArgumentTypeError("Use o formato: caminho,porcentagem (ex: ./data/train,80)")

# Extra check for resize
if 'resize' in args.process_type and args.resize_dim is None:
    print("Error: --resize-dim must be specified when --process-type includes 'resize'.")
    sys.exit(1)