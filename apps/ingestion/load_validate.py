from datetime import datetime
import json
import os
import pandas as pd
import shutil
from apps.database.database_operation import DatabaseOperation
from apps.core.logger import Logger


class LoadValidate:

    def __init__(self, run_id, data_path, mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'LoadValidate', mode)
        self.dbOperation = DatabaseOperation(self.run_id, self.data_path, mode)

    def read_values_from_schema(self, schema_file):
        try:
            self.logger.info('Start of reading values from schema...')
            with open(f'apps/database/{schema_file}.json', 'r') as f:
                dic = json.load(f)
                f.close()

            column_names = dic['ColName']
            number_of_columns = dic['NumberofColumns']
            self.logger.info('End of reading values from schema...')

        except ValueError:
            self.logger.exception(
                'ValueError raised while reading values from schema')
            raise ValueError

        except KeyError:
            self.logger.exception(
                'KeyError raised while reading values from schema')
            raise KeyError

        except Exception as e:
            self.logger.exception(
                'Exception raised when reading values from schema: %s' % e)
            raise e

        return column_names, number_of_columns

    def validate_column_length(self, number_of_columns):
        try:
            self.logger.info('Start of validating column length...')
            for file in os.listdir(self.data_path):
                csv = pd.read_csv(self.data_path + '/' + file)
                if csv.shape[1] != number_of_columns:
                    shutil.move(self.data_path + '/' + file,
                                self.data_path + '_rejects')
                    self.logger.info('Invalid column length :: %s' % file)

            self.logger.info('End of validating column length...')

        except OSError:
            self.logger.exception(
                'OSError raised when validating column length')
            raise OSError

        except Exception as e:
            self.logger.exception(
                'Exception raise when validating column length: %s' % e)
            raise e

    def validate_missing_values(self):
        # validates if a column has all values missing
        try:
            self.logger.info('Start of validating missing values...')
            for file in os.listdir(self.data_path):
                csv = pd.read_csv(self.data_path + '/' + file)
                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count += 1
                        shutil.move(self.data_path+'/' + file,
                                    self.data_path+'_rejects')
                        self.logger.info(
                            "All missing values in column :: %s" % file)
                        break

            self.logger.info('End of validating missing values...')
        except OSError:
            self.logger.exception(
                'OSError raised while validating missing values')
            raise OSError
        except Exception as e:
            self.logger.exception(
                'Exception raised while validating missing values: %s' % e)
            raise e

    def replace_missing_values(self):
        try:
            self.logger.info('Start of replacing missing values with NULL...')
            only_files = [f for f in os.listdir(self.data_path)]

            for file in only_files:
                csv = pd.read_csv(self.data_path + '/' + file)
                csv.fillna('NULL', inplace=True)
                csv.to_csv(self.data_path + '/' + file,
                           index=None, header=True)
                self.logger.info('%s: File transformed successully.' % file)

            self.logger.info('End of replacing missing values with NULL...')

        except Exception as e:
            self.logger.exception(
                'Exception raised while replacing missing values with NULL: %s' % e)
            raise e

    def archive_old_files(self):
        now = datetime.now()
        date = now.date()
        time = now.strftime('%H%M%S')

        def process_archival(file_grp):
            """
            file_grp (str): can be 'reject', 'validation', 'processed', 'result' files
            """

            self.logger.info('Start of archiving old %s files...' % file_grp)
            if file_grp == 'reject' or file_grp == 'result':
                source = f'{self.data_path}_{file_grp}s'
            else:
                source = f'{self.data_path}_{file_grp}'

            # create/initalize archive path if file_grp path exists
            if os.path.isdir(source):
                path = self.data_path + '_archive'
                if not os.path.isdir(path):
                    os.makedirs(path)
                dest = f'{path}/{file_grp}_{str(date)}_{str(time)}'

                files = os.listdir(source)
                for f in files:
                    if not os.path.isdir(dest):
                        os.makedirs(dest)
                    if f not in os.listdir(dest):
                        shutil.move(os.path.join(source, f), dest)
            self.logger.info('End of archiving old %s files...' % file_grp)

        try:
            # archive rejected files
            process_archival('reject')

            # archive validated files
            process_archival('validation')

            # archive processed files
            process_archival('processed')

            # archive result files
            process_archival('result')

        except Exception as e:
            self.logger.exception(
                'Exception raised while archiving old files: %s' % e)
            raise e

    def move_processed_files(self):
        try:
            self.logger.info('Start of moving processed files...')
            for file in os.listdir(self.data_path):
                shutil.move(f'{self.data_path}/{file}',
                            f'{self.data_path}_processed')
                self.logger.info('Moved the already processed file: %s' % file)

            self.logger.info('End of moving processed files...')

        except Exception as e:
            self.logger.exception(
                'Exception raised while moving processed files: %s' % e)
            raise e

    def validate_trainset(self):
        try:
            self.logger.info(
                'Start of data load, validation, and transformation for trainset')
            # archive old files
            self.archive_old_files()
            # extract values from training schema
            column_names, number_of_columns = self.read_values_from_schema(
                'schema_train')
            # validate column length in the file
            self.validate_column_length(number_of_columns)
            # validate if a column has all values missing
            self.validate_missing_values()
            # replace missing values with NULL
            self.replace_missing_values()
            # create database with given name (if existing, open the connection), create table with columns indicated in schema
            self.dbOperation.create_table(
                'training', 'training_raw_data_t', column_names)
            # insert all csv file data into the tale
            self.dbOperation.insert_data('training', 'training_raw_data_t')
            # export data in table to csv file
            self.dbOperation.export_csv('training', 'training_raw_data_t')
            # move processed files
            self.move_processed_files()
            self.logger.info(
                'End of data load, validation, and transformation')

        except Exception:
            self.logger.exception(
                'Encountered eror during data load, validation, and transformation for trainset')

    def validate_predictset(self):
        try:
            self.logger.info(
                'Start of data load, validation, and transformation for predictset')
            # archive old files
            self.archive_old_files()
            # extract values from training schema
            column_names, number_of_columns = self.read_values_from_schema(
                'schema_predict')
            # validate column length in the file
            self.validate_column_length(number_of_columns)
            # validate if a column has all values missing
            self.validate_missing_values()
            # replace missing values with NULL
            self.replace_missing_values()
            # create database with given name (if existing, open the connection), create table with columns indicated in schema
            self.dbOperation.create_table(
                'prediction', 'prediction_raw_data_t', column_names)
            # insert all csv file data into the tale
            self.dbOperation.insert_data('prediction', 'prediction_raw_data_t')
            # export data in table to csv file
            self.dbOperation.export_csv('prediction', 'prediction_raw_data_t')
            # move processed files
            self.move_processed_files()
            self.logger.info(
                'End of data load, validation, and transformation')

        except Exception:
            self.logger.exception(
                'Encountered eror during data load, validation, and transformation for predictset')
