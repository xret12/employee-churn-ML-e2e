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
        self.preProcess = Preprocessor(self.run_id, self.data_path, 'training')
        self.modelTuner = ModelTuner(self.run_id, self.data_path, 'training')
        self.FileOperation = FileOperation(self.run_id, self.data_path, 'training')
        self.cluster = KMeansCluster(self.run_id, self.data_path)


    def train_model(self):
        try:
            self.logger.info('Start of training model...')
            self.logger.info('Run_id: %s' % self.run_id)

            # load, validate, and transform dataset
            self.loadValidate.validate_trainset()

            # preprocess training data
            self.X, self.y = self.preProcess.preprocess_trainset()

            # create columns.json file
            columns = {'data_columns': [col for col in self.X.columns]}
            with open('apps/database/columns.json', 'w') as f:
                f.write(json.dumps(columns))

            # create clusters
            number_of_clusters = self.cluster.create_elbow_plot(self.X)

            # divide the data into clusters
            self.X = self.cluster.create_clusters(self.X, number_of_clusters)
            
            # attach labels (needed so that after clustering, we know the corresponding label for each cluster data)
            self.X['Labels'] = self.y

            # get unique clusters
            list_of_clusters = self.X['Cluster'].unique()

            # parse all clusters and look for best ML algo for each cluster
            for i in list_of_clusters:
                # filter using cluster name
                cluster_data = self.X[self.X['Cluster'] == i]
                # prep feature and label cols
                cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
                cluster_label = cluster_data['Labels']
                # split data into train, test
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label,
                                                                    test_size=0.2, random_state=0)
                best_model_name, best_model = self.modelTuner.get_best_model(x_train, y_train, x_test, y_test)
                # save model
                save_model = self.FileOperation.save_model(best_model, f'{best_model_name}{str(i)}')
            
            self.logger.info('End of training model...')

        except Exception as e:
            self.logger.exception('Training model unsuccessful: %s' % e)
            raise e