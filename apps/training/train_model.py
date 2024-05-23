import json
from sklearn.model_selection import train_test_split

from apps.core.logger import Logger
from apps.core.file_operation import FileOperation
from apps.ingestion.load_validate import LoadValidate
from apps.preprocess.preprocessor import Preprocessor
from apps.tuning.model_tuner import ModelTuner
from apps.tuning.cluster import KMeansCluster


class TrainModel:

    def __init__(self, run_id, data_path):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'TrainModel', 'training')
        self.loadValidate = LoadValidate(self.run_id, self.data_path, 'training')
        self.preProcess= Preprocessor(self.run_id, self.data_path, 'training')