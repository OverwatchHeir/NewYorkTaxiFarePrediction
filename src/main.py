from src.DataCleaning import DataCleaning
from src.MLModelEvaluation import MLModelEvaluation

if __name__ == '__main__':

    # COMMENT IF ALREADY CLEANED
    # cleaner = DataCleaning()
    # cleaner.clean()
    # cleaner.plotOnMap()

    evaluation = MLModelEvaluation()
    evaluation.mlModelEvaluation()

