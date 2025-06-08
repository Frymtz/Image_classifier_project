#from dataset import Dataset
from model import *
from utils import Logger
from utils import checks as ch

def main(args):
    log = Logger(name="main", level=10)
    log.info("Stating the program...")
#---------------------------------------------------------------------------------------------#
#Check parsers
    # Perform all argument verifications
    try:
        train_percent, validation_percent, test_percent=ch.verify_all_args(args)
    except Exception as e:
        log = Logger(name="main.checks", level=10)
        log.error(f"Argument verification failed: {e}")
        raise
    log.info("All arguments verified successfully.")

    #print(f"Train: {train_percent}%, Validation: {validation_percent}%, Test: {test_percent}%")
#---------------------------------------------------------------------------------------------#
#Dataset
    # try:
    #     #dataset = Dataset(args.train, args.validation, args.test, train_percent, validation_percent, test_percent)
    #     log.info("Dataset initialized successfully.")
    # except Exception as e:
    #     log.error(f"Failed to initialize dataset: {e}")
    #     raise

if __name__ == "__main__":
    main()