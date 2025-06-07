from dataset import *
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
#---------------------------------------------------------------------------------------------#
    print(train_percent, validation_percent, test_percent)

if __name__ == "__main__":
    main()