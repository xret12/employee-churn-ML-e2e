import json
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from apps.core.logger import Logger


class Preprocessor:

    def __init__(self, run_id, data_path, mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'Preprocessor', mode)

    def get_data(self):
        try:
            self.logger.info('Start of reading dataset...')
            self.data = pd.read_csv(
                f'{self.data_path}_validation/InputFile.csv')
            self.logger.info('End of reading dataset....')
            return self.data

        except Exception as e:
            self.logger.exception(
                'Exception rasied while reading dataset: %s' % e)
            raise e

    def drop_columns(self, data, columns):
        self.data = data
        self.columns = columns

        try:
            self.logger.info('Start of dropping columns...')
            self.useful_data = self.data.drop(
                labels=self.columns, axis=1)  # drop the specified columns
            self.logger.info('End of dropping columns...')
            return self.useful_data

        except Exception as e:
            self.logger.exception(
                'Exception raised while dropping columns: %s' % e)
            raise e

    def is_null_present(self, data):
        try:
            self.logger.info('Start of finding missing values...')

            self.null_counts = data.isna().sum()
            self.null_present = any(x > 0 for x in self.null_counts)

            if self.null_present:
                df_with_null = pd.DataFrame()
                df_with_null['columns'] = data.columns
                df_with_null['missing value counts'] = np.asarray(
                    self.null_counts)
                df_with_null.to_csv(
                    f'{self.data_path}_validation/null_values.csv')

            self.logger.info('End of finding missing values...')
            return self.null_present

        except Exception as e:
            self.logger.exception(
                'Exception raised while finding missing values: %s' % e)
            raise e

    def impute_missing_values(self, data):
        self.data = data

        try:
            self.logger.info('Start of imputing missing values...')
            imputer = KNNImputer(n_neighbors=3)
            # impute missing values
            arr = imputer.fit_transform(self.data)
            # convert arr into df
            self.new_data = pd.DataFrame(data=arr, columns=self.data.columns)

            self.logger.info('End of imputing missing values...')

            return self.new_data

        except Exception as e:
            self.logger.exception(
                'Exception raised while imputing missing values: %s' % e)
            raise e

    def encode_features(self, data):
        try:
            self.logger.info('Start of feature encoding...')

            # create a clone df with only object dtype columns
            self.new_data = data.select_dtypes(include=['object']).copy()

            # one-hot encode categorical columns
            for col in self.new_data.columns:
                self.new_data = pd.get_dummies(self.new_data, columns=[col],
                                               prefix=[col], drop_first=True)

            self.logger.info('End of feature encoding...')
            return self.new_data

        except Exception as e:
            self.logger.exception(
                'Exception raised while encoding features: %s' % e)
            raise e

    def split_features_label(self, data, label_name):
        self.data = data

        try:
            self.logger.info('Start of splitting features and label')

            # separate features and label
            self.X = self.data.drop(labels=label_name, axis=1)
            self.y = self.data[label_name]

            self.logger.info('End of splitting features and label')
            return self.X, self.y

        except Exception as e:
            self.logger.exception(
                'Exception raised while splitting features and label: %s' % e)
            raise e

    def build_final_predictset(self, data):
        try:
            self.logger.info('Start of building final predictset...')

            with open('apps/database/columns.json', 'r') as f:
                data_columns = json.load(f)['data_columns']
                f.close()

            df = pd.DataFrame(data=None, columns=data_columns)
            df_new = pd.concat([df, data], ignore_index=True)
            data_new = df_new.fillna(0)

            self.logger.info('End of building final predictset')
            return data_new

        except ValueError:
            self.logger.exception(
                'ValueError raised while building final predictset')
            raise ValueError
        except KeyError:
            self.logger.exception(
                'KeyError raised while building final predictset')
            raise KeyError
        except Exception as e:
            self.logger.exception(
                'Exception raised while building final predictset: %s' % e)
            raise e

    def preprocess_trainset(self):
        try:
            self.logger.info('Start of preprocessing training data...')
            # get data into pandas df
            data = self.get_data()
            # drop unwanted columns
            data = self.drop_columns(data, ['empid'])
            # handle label encoding
            cat_df = self.encode_features(data)
            data = pd.concat([data, cat_df], axis=1)
            # drop categorical column
            data = self.drop_columns(data, ['salary'])
            # check if missing values are present in data set
            is_null_present = self.is_null_present(data)
            # if missing values exist, impute
            if is_null_present:
                data = self.impute_missing_values(data)
            # create separate features and labels
            self.X, self.y = self.split_features_label(data, label_name='left')
            self.logger.info('End of preprocessing training data...')
            return self.X, self.y

        except Exception as e:
            self.logger.exception(
                'Exception raised while preprocessing training data: %s' % e)
            raise e

    def preprocess_predictset(self):
        try:
            self.logger.info('Start of preprocessing prediction data...')
            # get data into pandas df
            data = self.get_data()
            # drop unwanted columns
            # data = self.drop_columns(data, ['empid'])
            # handle label encoding
            cat_df = self.encode_features(data)
            data = pd.concat([data, cat_df], axis=1)
            # drop categorical column
            data = self.drop_columns(data, ['salary'])
            # check if missing values are present in data set
            is_null_present = self.is_null_present(data)
            # if missing values exist, impute
            if is_null_present:
                data = self.impute_missing_values(data)

            data = self.build_final_predictset(data)
            self.logger.info('End of preprocessing prediction data...')
            return data

        except Exception as e:
            self.logger.exception(
                'Exception raised while preprocessing prediction data: %s' % e)
            raise e

    def preprocess_single_predict(self, data):
        try:
            self.logger.info('Start of preprocessing for single predict...')
            cat_df = self.encode_features(data)
            data = pd.concat([data, cat_df], axis=1)
            # drop categorical column
            data = self.drop_columns(data, ['salary'])
            # check if missing values are present in the data set
            is_null_present = self.is_null_present(data)
            # if missing values are there, replace them appropriately.
            if (is_null_present):
                data = self.impute_missing_values(
                    data)  # missing value imputation

            data = self.build_final_predictset(data)
            self.logger.info('End of preprocessing for single predict...')
            return data

        except Exception:
            self.logger.exception('Unsuccessful End of Preprocessing...')
            raise Exception
