from kneed import KneeLocator
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from apps.core.file_operation import FileOperation
from apps.core.logger import Logger
from apps.ingestion.load_validate import LoadValidate
from apps.preprocess.preprocessor import Preprocessor
from apps.tuning.model_tuner import ModelTuner


class KMeansCluster:

    def __init__(self, run_id, data_path):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'KMeansCluster', 'training')
        self.fileOperation = FileOperation(
            self.run_id, self.data_path, 'training')

    def create_elbow_plot(self, data):
        wcss = []  # initialize empty list --within cluster sum of errors
        try:
            self.logger.info('Start of elbow plotting...')
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, random_state=0)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)

            # plot WVSS vs number of clusters graph
            plt.plot(range(1, 11), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS')

            save_path = 'app/models'
            plt.savefig(f'{save_path}/kmeans_elbow.png')

            # find the optimum cluster count
            self.kn = KneeLocator(range(1, 11), wcss,
                                  curve='convex', direction='decreasing')
            self.logger.info('Optimum number of clusters: %s' %
                             str(self.kn.knee))
            self.logger.info('End of elbow plotting...')
            return self.kn.knee

        except Exception as e:
            self.logger.exception(
                'Exception raised while elbow plotting: %s' % e)
            raise e

    def create_clusters(self, data, number_of_clusters):
        self.data = data
        try:
            self.logger.info('Start of creating clusters...')
            self.kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)
            # divide data into clusters
            self.y_kmeans = self.kmeans.fit_predict(data)
            # save model
            self.saveModel = self.fileOperation.save_model(
                self.kmeans, 'KMeans')

            # create new column in dataset for assigning cluster info
            self.data['Cluster'] = self.y_kmeans

            self.logger.info('Successfully created %s cluster' %
                             str(self.kn.knee))
            self.logger.info('End of creating clusters...')
            return self.data

        except Exception as e:
            self.logger.exception(
                'Exception raised while creating clusters: %s' % e)
            raise e
