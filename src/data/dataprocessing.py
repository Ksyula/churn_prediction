import pandas as pd
import numpy as np
import os.path as op

binary_dict = {'Yes': 1, 'No': 0}
gender_dict = {'Female': 1, 'Male': 0}
root_path = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))


class Dataprocesser():
    """
    class for processing a raw dataset to a precessed data for model training
    """
    def __init__(self, raw_data_path: str, data_processed_path: str):
        """
        initiate Data processing object to convert raw data to processed data
        :param raw_data_path: str
        absolute path to a raw dataset
        """
        self.raw_data_path = op.join(root_path, raw_data_path)
        self.data_processed_path = op.join(root_path, data_processed_path)
        self.yes_no_cols = ["PaperlessBilling", "PhoneService", "Dependents", "Partner", "Churn"]
        self.categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                 'Contract', 'PaymentMethod']
        self.imputation_cols = "TotalCharges"
        self.gender = "gender"
        self.index = "customerID"

    def read_raw_data(self):
        """
        read the raw dataset from local path
        """
        raw_data = pd.read_csv(self.raw_data_path)
        return raw_data

    def impute_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        impute missing values according to mean strategy
        :param df: pd.DataFrame
        raw dataset
        """
        df.loc[:, self.imputation_cols] = pd.to_numeric(df[self.imputation_cols], errors='coerce')
        df.loc[:, self.imputation_cols] = df[self.imputation_cols].fillna(value=np.mean(df[self.imputation_cols]))
        return df

    def convert_categorical_to_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        transform columns with Yes/No values to 1/0
        :param df: pd.DataFrame
        raw dataset
        """
        df.loc[:, self.yes_no_cols] = df[self.yes_no_cols].stack().map(binary_dict).unstack()
        return df

    def convert_gender_to_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        transform gender to binary
        :param df: pd.DataFrame
        raw dataset
        """
        df.loc[:, self.gender] = df.gender.map(gender_dict)
        return df

    def convert_categorical_to_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        convert categorical columns with cardinality > 2 to dummies
        :param df: pd.DataFrame
        raw dataset
        """
        df = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)
        return df

    def process_data(self):
        """
        process raw data
        """
        df = self.read_raw_data()
        df = self.impute_mean(df)
        df = self.convert_categorical_to_binary(df)
        df = self.convert_gender_to_binary(df)
        df = self.convert_categorical_to_dummies(df)
        df.set_index(self.index, inplace=True)
        self.write_processed_dataset(df)

    def write_processed_dataset(self, df: pd.DataFrame):
        df.to_csv(self.data_processed_path, index=False)


