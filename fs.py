import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
from scipy.stats import pearsonr
from loge_code import setup_logging
logger=setup_logging('fs')
from sklearn.feature_selection import VarianceThreshold
def select_features(X_train, X_test, y_train, y_test):
    try:
        #constant Technique
        logger.info(f"Before constant Technique X_train columns and shape : {X_train.columns} : {X_train.shape}")
        logger.info(f"Before constant Technique X_test columns and shape : {X_test.columns} : {X_test.shape}")
        con_obj=VarianceThreshold(threshold=0.0)
        con_obj.fit(X_train)
        logger.info(f"columns to be remove:{X_train.columns[~con_obj.get_support()]}")
        X_train=X_train.drop(['fbs_yeo_trim'],axis=1)
        X_test=X_test.drop(['fbs_yeo_trim'],axis=1)
        logger.info(f"After constant Technique X_train columns and shape : {X_train.columns} : {X_train.shape}")
        logger.info(f"After constant Technique X_test columns and shape : {X_test.columns} : {X_test.shape}")

        # quasi constant Technique
        logger.info(f"Before quasi constant Technique X_train columns and shape : {X_train.columns} : {X_train.shape}")
        logger.info(f"Before quasi constant Technique X_test columns and shape : {X_test.columns} : {X_test.shape}")
        qua_obj = VarianceThreshold(threshold=0.1)
        qua_obj.fit(X_train)
        logger.info(f"columns to be remove:{X_train.columns[~qua_obj.get_support()]}")
        X_train = X_train.drop(['trestbps_yeo_trim', 'chol_yeo_trim', 'exang_yeo_trim', 'ca_yeo_trim'], axis=1)
        X_test = X_test.drop(['trestbps_yeo_trim', 'chol_yeo_trim', 'exang_yeo_trim', 'ca_yeo_trim'], axis=1)
        logger.info(f"After quasi constant Technique X_train columns and shape : {X_train.columns} : {X_train.shape}")
        logger.info(f"After quasi constant Technique X_test columns and shape : {X_test.columns} : {X_test.shape}")

        #correlation with Hypothesis Testing
        logger.info(f"Before Hypothesis Testing X_train columns and shape : {X_train.columns} : {X_train.shape}")
        logger.info(f"Before Hypothesis Testing X_test columns and shape : {X_test.columns} : {X_test.shape}")
        corr_p_values = []
        for i in X_train.columns:
            values = pearsonr(X_train[i] , y_train)
            corr_p_values.append(values)
        corr_p_values = np.array(corr_p_values)
        p_values = corr_p_values[:, 1]
        c=0
        for i in p_values:
            if i>0.05:
                logger.info(f'{X_train.columns[c]}: {i}')
            c+=1
        X_train=X_train.drop(['restecg_yeo_trim'],axis=1)
        X_test=X_test.drop(['restecg_yeo_trim'],axis=1)
        logger.info(f"After Hypothesis Testing X_train columns and shape : {X_train.columns} : {X_train.shape}")
        logger.info(f"After Hypothesis Testing X_test columns and shape : {X_test.columns} : {X_test.shape}")
        return X_train,X_test

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

