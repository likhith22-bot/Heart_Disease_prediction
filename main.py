"""
Heart Disease Prediction using Machine Learning.
Performs data preprocessing, feature selection, SMOTE balancing, scaling, and Gaussian Naive Bayes model tuning.
Saves the trained scaler and final model for making predictions on new patient data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns
import warnings
import pickle

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from varibale_transformation import check,yeo_johnson,trimming
from training_models import common
from fs import select_features
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from hyperparameter_tuning import grid_search_cv
from loge_code import setup_logging
logger = setup_logging('main')
class Heart_Disease_Pridiction:
    def __init__(self,path):
        try:
            self.path = path
            self.df=pd.read_csv(self.path)
            logger.info(f"The Number of rows and columns was : {self.df.shape}")
            logger.info(f"NUll values in the values  : {self.df.isnull().sum()}")
            self.X = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            logger.info(f"Training dataset size : \n {self.X_train.shape}  => {self.y_train.shape}")
            logger.info(f"Testing dataset size : \n {self.X_test.shape} => {self.y_test.shape}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
    def varibale_transform(self):
        try:
            logger.info(f"Before yeojohnson transformation columns : \n {self.X_train.columns}")
            logger.info(f"Before yeojohnson transformation columns : \n {self.X_test.columns}")
            self.X_train,self.X_test=yeo_johnson(self.X_train,self.X_test)
            logger.info(f"After yeojohnson transformation columns : \n {self.X_train.columns}")
            logger.info(f"After yeojohnson transformation columns : \n {self.X_test.columns}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
    def outliers(self):
        try:
            logger.info(f"before applying trimming X_train columns : \n {self.X_train.columns}")
            logger.info(f"before applying trimming X_test columns : \n {self.X_test.columns}")
            self.X_train,self.X_test=trimming(self.X_train,self.X_test)
            logger.info(f"After applying trimming X_train columns : \n {self.X_train.columns}")
            logger.info(f"After applying trimming X_test columns : \n {self.X_test.columns}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
    def feature_selection(self):
        try:
            logger.info(f"Before applying  feature selection X_train columns and shape : {self.X_train.columns} : {self.X_train.shape}")
            logger.info(f"Before applying feature selection  X_test columns and shape : {self.X_test.columns} : {self.X_test.shape}")
            self.X_train,self.X_test=select_features(self.X_train,self.X_test,self.y_train,self.y_test)
            logger.info(f"After applying  feature selection X_train columns and shape : {self.X_train.columns} : {self.X_train.shape}")
            logger.info(f"After applying feature selection  X_test columns and shape : {self.X_test.columns} : {self.X_test.shape}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
    def data_balancing(self):
        try:
            logger.info(f"Before data balancing : {self.y_train.shape}")
            logger.info(f"Number of Rows for 1 :  {sum(self.y_train == 1)}")
            logger.info(f"Number of Rows for 0 :  {sum(self.y_train == 0)}")
            s_obj = SMOTE(random_state=42)
            self.X_train,self.y_train=s_obj.fit_resample(self.X_train,self.y_train)
            logger.info(f"After data balancing : {self.y_train.shape}")
            logger.info(f"Number of Rows for 1 :  {sum(self.y_train == 1)}")
            logger.info(f"Number of Rows for 0 :  {sum(self.y_train == 0)}")

            #scaling down the values
            global sc
            sc=StandardScaler()
            sc.fit(self.X_train)
            self.X_train=pd.DataFrame(sc.transform(self.X_train))
            self.X_test=pd.DataFrame(sc.transform(self.X_test))
            with open('scaling.pkl', 'wb') as f:
                pickle.dump(sc,f)
            logger.info("--------------------------------Final data sets-------------------------------")
            logger.info(f"===========Final Train Data============================= : \n : {self.X_train.shape} : \n : {self.X_train.columns} : \n : {self.X_train.head(10)} : \n : {self.X_train.isnull().sum()}")
            logger.info(f"===========Final Test Data============================= : \n : {self.X_test.shape} : \n : {self.X_test.columns} : \n : {self.X_test.head(10)} : \n : {self.X_test.isnull().sum()}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
    def train_all_model(self):
        try:
            pass
            #common(self.X_train,self.y_train,self.X_test,self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
    def select_parameters(self):
        try:
           model= grid_search_cv(self.X_train, self.y_train, self.X_test, self.y_test)
           print(model.predict(sc.transform()))
           print(accuracy_score(self.y_test,model.predict(sc.transform(self.X_test))))

           with open('final_model.pkl', 'wb') as f:
               pickle.dump(model,f)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
    def sample_pridiction(self):
        try:
            val=np.array([[41,0,1,172,1.4,0,2]])
            with open('scaling.pkl', 'rb') as f:
                s =pickle.load(f)
            with open('final_model.pkl', 'rb') as m:
                reg=pickle.load(m)
            logger.info(f"After sample pridiction : {reg.predict(s.transform(val))}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in line no : {er_line.tb_lineno} due to : {er_msg}')
if __name__=="__main__":
    try:
        obj=Heart_Disease_Pridiction("heart.csv")
        obj.varibale_transform()
        obj.outliers()
        obj.feature_selection()
        obj.data_balancing()
        obj.train_all_model()
        obj.select_parameters()
        obj.sample_pridiction()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in line no : {er_line.tb_lineno} due to : {er_msg}')


