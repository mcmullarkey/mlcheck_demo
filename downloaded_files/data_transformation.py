import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException

from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):
        '''
        This function is responsible for Data Transformation
        '''
        try:
            # Define the Numerical Columns
            numerical_columns=["writing_score","reading_score"]
            # Define the Categorical Column
            categorical_coumns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            #Setup the numerical pipeline steps
            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                    
                ]
            )

            #Setup the Csategorical pipeline steps
            cat_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("on_hot_encoder", OneHotEncoder()),
                    ("scaler" , StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Numerical Columns : {numerical_columns}")
            logging.info(f"Categorical Columns :{categorical_coumns}")

            # Pass the pipeline and columns as parameter to column transformer.
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_coumns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read Train & Test Data Completed")
            logging.info("obtaining the preprocessing object")

            # Call the get_data_transformation_object method and get the preprocessor object (Column Tranformer)
            preprocessing_object= self.get_data_transformation_object()
            target_column_name="math_score"
            numerical_column=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying pre processing object on training dataframe & testiing data frame.")
            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.fit_transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing Object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )

            return(
                train_arr,
                test_arr,
                #self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)