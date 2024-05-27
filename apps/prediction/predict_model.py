import pandas as pd
from apps.core.logger import Logger
from apps.core.file_operation import FileOperation
from apps.ingestion.load_validate import LoadValidate
from apps.preprocess.preprocessor import Preprocessor


class PredictModel:

    def __init__(self, run_id, data_path):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'PredictModel', 'prediction')
        self.loadValidate = LoadValidate(
            self.run_id, self.data_path, 'prediction')
        self.preProcess = Preprocessor(
            self.run_id, self.data_path, 'prediction')
        self.fileOperation = FileOperation(
            self.run_id, self.data_path, 'prediction')

    def batch_predict_from_model(self):
        try:
            self.logger.info('Start of prediction...')
            self.logger.info('Run ID: %s' % self.run_id)

            # validate and transform data
            self.loadValidate.validate_predictset()

            # preprocess data
            self.X = self.preProcess.preprocess_predictset()

            # load model
            kmeans = self.fileOperation.load_model('KMeans')

            # set clusters
            clusters = kmeans.predict(self.X.drop(['empid'], axis=1))
            self.X['Cluster'] = clusters
            clusters = self.X['Cluster'].unique()

            for i in clusters:
                self.logger.info('Cluster loop started')
                cluster_data = self.X[self.X['Cluster'] == i]
                cluster_data_new = cluster_data.drop(
                    ['empid', 'Cluster'], axis=1)
                model_name = self.fileOperation.correct_model(i)
                model = self.fileOperation.load_model(model_name)
                y_predicted = model.predict(cluster_data_new)
                result = pd.DataFrame({'EmpId': cluster_data['empid'],
                                       'Prediction': y_predicted})
                result.to_csv(f'{self.data_path}_results/Prediction.csv',
                              header=True, mode='a+', index=False)

            self.logger.info('End of prediction...')

        except Exception as e:
            self.logger.exception('Exception raised while predicting: %s' % e)
            raise e

    def single_predict_from_model(self, data):
        try:
            self.logger.info('Start of prediction...')
            self.logger.info('Run ID: %s' % self.run_id)
            # preprocess data
            self.X = self.preProcess.preprocess_single_predict(data)
            # load model
            kmeans = self.fileOperation.load_model('KMeans')
            # select clusters
            clusters = kmeans.predict(self.X.drop(['empid'], axis=1))
            self.X['Cluster'] = clusters
            clusters = self.X['Cluster'].unique()

            for i in clusters:
                self.logger.info('clusters loop started')
                cluster_data = self.X[self.X['Cluster'] == i]
                cluster_data_new = cluster_data.drop(
                    ['empid', 'Cluster'], axis=1)
                model_name = self.fileOperation.correct_model(i)
                model = self.fileOperation.load_model(model_name)
                self.logger.info('Shape of Data ' +
                                 str(cluster_data_new.shape))
                self.logger.info('Shape of Data ' +
                                 str(cluster_data_new.info()))
                y_predicted = model.predict(cluster_data_new)

                self.logger.info('Output : ' + str(y_predicted))
                self.logger.info('End of prediction...')

                return int(y_predicted[0])

        except Exception as e:
            self.logger.exception('Exception raised while predicting: %s' % e)
            raise e
