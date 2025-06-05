from dataset import *
from model import *
from utils import Logger

def main(args):

#---------------------------------------------------------------------------------------------#
# #Check parsers

    print("Parsing arguments...")
    print("Train dataset path:", args.train[0])
    print("Train labels path:", args.train[1])
    print("Train percentage:", args.train[2])

    print("Validation dataset path:", args.validation[0])
    print("Validation labels path:", args.validation[1])    
    print("Validation percentage:", args.validation[2])

    print("Test dataset path:", args.test[0])
    print("Test labels path:", args.test[1])
    print("Test percentage:", args.test[2])

    print("Resize dimensions:", args.resize if args.resize else "Not specified")
    print("Feature extraction techniques:", args.features if args.features else "Not specified")  

    print("Result types:", args.result_type if args.result_type else "Not specified")




#---------------------------------------------------------------------------------------------#







    log = Logger(name="main", level=10)  
    log.info("Iniciando o programa...")
    log.warning("Isso é um aviso!")
    log.error("Ocorreu um erro!")

if __name__ == "__main__":
    main()