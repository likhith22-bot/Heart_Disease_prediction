import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from loge_code import setup_logging
logger=setup_logging('hyperparameter_tuning')



def grid_search_cv(X_train, y_train, X_test, y_test):
    try:
        parameters={'var_smoothing': [1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]}
        nb=GaussianNB()
        grid_obj=GridSearchCV(estimator=nb, param_grid=parameters,scoring='accuracy', cv=10)
        grid_obj.fit(X_train, y_train)
        logger.info(f"best params : {grid_obj.best_params_}")
        nb_reg=GaussianNB(var_smoothing=1e-12)
        nb_reg.fit(X_train, y_train)
        logger.info(f"Test Data Accuracy : {grid_obj.best_score_}")

        return nb_reg

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

