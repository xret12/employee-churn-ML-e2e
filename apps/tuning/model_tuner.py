from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from apps.core.logger import Logger
from sklearn.metrics import r2_score


class ModelTuner:

    def __init__(self, run_id, data_path, mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'ModelTuner', mode)
        self.rfc = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')

    
    def find_best_params_randomforest(self, train_x, train_y):
        try:
            self.logger.info('Start of finding the best params for Random Forest...')
            # intialize parameter grid
            self.param_grid = {
                'n_estimators': [10, 50, 100, 130],
                'criterion': ['gini', 'entropy'],
                'max_depth': range(2, 4, 1),
                'max_features': ['sqrt', 'log2']
            }

            # use grid search for hyperparameter tuning
            self.grid = GridSearchCV(estimator=self.rfc, param_grid=self.param_grid, cv=5)
            # find best parameters
            self.grid.fit(train_x, train_y)

            # extract best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # create new model with best parameters
            self.rfc = RandomForestClassifier(n_estimators=self.n_estimators,
                                              criterion=self.criterion,
                                              max_depth=self.max_depth,
                                              max_features=self.max_features
            )

            # train the new model
            self.rfc.fit(train_x, train_y)
            
            self.logger.info('Random Forest Best Params: %s' % str(self.grid.best_params_))
            self.logger.info('End of finding best params for Random Forest...')

            return self.rfc
        
        except Exception as e:
            self.logger.exception('Exception raised while finding best params for Random Forest: %s' % e)
            raise e
        
    
    def find_best_params_xgboost(self, train_x, train_y):
            try:
                self.logger.info('Start of finding best params for XGBoost...')
                # initialize params grid
                self.param_grid = {
                    'learning_rate': [0.5, 0.1, 0.01, 0.001],
                    'max_depth': [3, 5, 10, 20],
                    'n_estimators': [10, 50, 100, 200]
                }

                # use grid search
                self.grid = GridSearchCV(estimator=self.xgb, param_grid=self.param_grid, cv=5)
                self.grid.fit(train_x, train_y)

                # extract best parameters
                self.learnig_rate = self.grid.best_params_['learning_rate']
                self.max_depth = self.grid.best_params_['max_depth']
                self.n_estimators = self.grid.best_params_['n_estimators']

                # create new model with best parameters
                self.xgb = XGBClassifier(objective='binary:logistic',
                                        learning_rate=self.learnig_rate,
                                        max_depth=self.max_depth,
                                        n_estimators=self.n_estimators
            )
                # train new model
                self.xgb.fit(train_x, train_y)
            
                self.logger.info('XGBoost best params: %s' % str(self.grid.best_params_))
                self.logger.info('End of finding best params for XGBoost...')

                return self.xgb

            except Exception as e:
                self.logger.exception('Exception raised while finding best params for XGBoost: %s' % e)
                raise e
            
    
    def get_best_model(self, train_x, train_y, test_x, test_y):
        try:
            self.logger.info('Start of finding best model')
            self.xgboost = self.find_best_params_xgboost(train_x, train_y)
            # predict using xgboost
            self.prediction_xgboost = self.xgboost.predict(test_x)
        
            if len(test_y.unique()) == 1: # if there is only one label in y, then roc_auc_score returns error. Will use accuracy instead
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger.info('Accuracy for XGBoost: %s' % str(self.xgboost_score))
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost)
                self.logger.info('AUC for XGBoost: %s' % str(self.xgboost_score))

            # create best model for Random Forest
            self.random_forest = self.find_best_params_randomforest(train_x, train_y)
            self.prediction_random_forest = self.random_forest.predict(test_x)

            if len(test_y.unique()) == 1: # if there is only one label in y, then roc_auc_score returns error. Will use accuracy instead
                self.random_forest_score = accuracy_score(test_y, self.prediction_random_forest)
                self.logger.info('Accuracy for Random Forest: %s' % str(self.random_forest_score))
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest)
                self.logger.info('AUC for Random Forest: %s' % str(self.random_forest_score))

            # compare the two models
            self.logger.info('End of finding the best model...')
            if self.xgboost_score > self.random_forest_score:
                return 'XGBoost', self.xgboost
            else:
                return 'RandomForest', self.random_forest
            
        except Exception as e:
            self.logger.exception('Exception raised while finding the best model: %s' % e)
            raise e
        
         