from datetime import datetime
import logging
import os

class Logger:

    def __init__(self, run_id, log_module, log_file_name):
        self.logger = logging.getLogger(str(log_module) + '_' + str(run_id))
        self.logger.setLevel(logging.DEBUG)

        if log_file_name == 'training':
            path = 'logs/training_logs'
            os.makedirs(path, exist_ok=True)
            file_handler = logging.FileHandler(path + '/train_log_' + str(run_id) + '.log')
        else:
            path = 'logs/prediction_logs'
            os.makedirs(path, exist_ok=True)
            file_handler = logging.FileHandler(path + '/predict_log_' + str(run_id) + '.log')

        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message):
        self.logger.info(message)

    def exception(self, message):
        self.logger.exception(message)