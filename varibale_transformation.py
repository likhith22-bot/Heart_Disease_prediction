import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import yeojohnson
from loge_code import setup_logging
logger = setup_logging('Varibale-Transformation')
def check(X):
    try:
        for i in X.columns:
            plt.figure(figsize = (5,3))
            plt.title(f"Box plot of {i}")
            sns.boxplot(X[i])
            plt.show()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
def yeo_johnson(X_train, X_test):
    try:
        logger.info(f"before applying yeojohnson X_train columns : {X_train.columns} ")
        logger.info(f"before applying yeojohnson X_test columns : {X_test.columns}")

        for i in X_train.columns:
            X_train[i+"_yeo"],lam = yeojohnson(X_train[i])
            X_test[i+"_yeo"],lam = yeojohnson(X_test[i])
            X_train=X_train.drop([i],axis=1)
            X_test=X_test.drop([i],axis=1)
        logger.info(f"After applying yeojohnson X_train columns : \n {X_train.columns}")
        logger.info(f"After applying yeojohnson X_test columns : \n {X_test.columns}")
        return X_train, X_test

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
def trimming(X_train, X_test):
    try:
        logger.info(f"before applying trimming X_train columns : \n {X_train.columns}")
        logger.info(f"before applying trimming X_test columns : \n {X_test.columns}")
        for i in X_train.columns:
            iqr = X_train[i].quantile(0.75) - X_train[i].quantile(0.25)
            upper_limit = X_train[i].quantile(0.75) + (1.5 * iqr)
            lower_limit = X_train[i].quantile(0.25) - (1.5 * iqr)

            X_train[i + "_trim"] = np.where(X_train[i] > upper_limit, upper_limit,
                                                np.where(X_train[i] < lower_limit, lower_limit,
                                                         X_train[i]))

            X_test[i + "_trim"] = np.where(X_test[i] > upper_limit, upper_limit,
                                               np.where(X_test[i] < lower_limit, lower_limit,
                                                        X_test[i]))

            X_train = X_train.drop([i], axis=1)
            X_test = X_test.drop([i], axis=1)
            logger.info(f"After applying trimming X_train columns : \n {X_train.columns}")
            logger.info(f"After applying trimming X_test columns : \n {X_test.columns}")
        return X_train, X_test
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")