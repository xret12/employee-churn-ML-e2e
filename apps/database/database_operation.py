import sqlite3
import csv
import os
import shutil

from apps.core.logger import Logger


class DatabaseOperation:

    def __init__(self, run_id, data_path, mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'DatabaseOperation', mode)

    def database_connection(self, database_name):
        try:
            conn = sqlite3.connect('apps/database/' + database_name + '.db')
            self.logger.info('Opened %s database successfully' % database_name)

        except ConnectionError as ce:
            self.logger.info('Error while connecting to database: %s' % ce)
            raise ce

        return conn

    def create_table(self, database_name, table_name, column_names):
        try:
            self.logger.info('Start of creating table...')
            conn = self.database_connection(database_name)

            if database_name == 'prediction':
                conn.execute(f"DROP TABLE IF EXISTS '{table_name}';")

            c = conn.cursor()
            c.execute(
                f"SELECT count(name) FROM sqlite_master WHERE type = 'table' AND name = '{table_name}'")
            if c.fetchone()[0] == 1:
                conn.close()
                self.logger.info('Tables created successfully')
                self.logger.info(
                    'Closed %s database successfilly' % table_name)
            else:
                for key in column_names.keys():
                    type = column_names[key]
                    # code in try block assumes that table exists, if not: table will be created
                    try:
                        c.execute(
                            f"ALTER TABLE {table_name} ADD COLUMN {key} {type}")
                        self.logger.info(
                            "ALTER TABLE %s ADD COLUMN" % table_name)
                    except:
                        c.execute(f"CREATE TABLE {table_name} ({key} {type})")
                        self.logger.info("CREATE TABLE %s %s" %
                                         (table_name, key))
                conn.commit()
                conn.close()
            self.logger.info("End of creating table...")
        except Exception as e:
            self.logger.exception(
                "Exception raised while creating table: %s" % e)
            raise e

    def insert_data(self, database_name, table_name):
        conn = self.database_connection(database_name)
        good_data_path = self.data_path
        bad_data_path = self.data_path + '_rejects'
        only_files = [f for f in os.listdir(good_data_path)]
        self.logger.info('Start of inserting data into table...')

        for file in only_files:
            try:
                with open(good_data_path + '/' + file, 'r') as f:
                    next(f)  # skips header line (first line)
                    reader = csv.reader(f, delimiter=',')
                    for line in enumerate(reader):
                        # self.logger.info(" %s: nu!!" % line[1])
                        to_db = ''
                        for list_ in (line[1]):
                            try:
                                to_db = to_db + ",'"+list_+"'"
                            except Exception as e:
                                raise e
                        # self.logger.info(" %s: list_!!" % to_db.lstrip(','))
                        to_db = to_db.lstrip(',')
                        conn.execute("INSERT INTO "+table_name +
                                     " values ({values})".format(values=(to_db)))
                        conn.commit()

            except Exception as e:
                conn.rollback()
                self.logger.exception(
                    'Exception raised while inserting data into table: %s' % e)
                shutil.move(good_data_path + '/' + file, bad_data_path)
                conn.close()
        conn.close()
        self.logger.info('End of inserting data into table...')

    def export_csv(self, database_name, table_name):
        self.file_from_db = self.data_path + str('_validation/')
        self.file_name = 'InputFile.csv'

        try:
            self.logger.info('Start of exporting data into CSV...')
            conn = self.database_connection(database_name)
            c = conn.cursor()
            c.execute(f"SELECT * from {table_name}")
            results = c.fetchall()

            # get headers of csv file
            headers = [i[0] for i in c.description]

            # make the csv output directory
            if not os.path.isdir(self.file_from_db):
                os.makedirs(self.file_from_db)

            # open csv for writing
            csv_file = csv.writer(open(self.file_from_db + self.file_name, 'w', newline=''),
                                  delimiter=',', lineterminator='\r\n',
                                  quoting=csv.QUOTE_ALL, escapechar='\\')

            # add headers and data to csv file
            csv_file.writerow(headers)
            csv_file.writerows(results)
            self.logger.info('End of exporting data to CSV...')

        except Exception as e:
            self.logger.exception(
                'Exception raised while exporting data to CSV: %s' % e)
