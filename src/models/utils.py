import numpy as np
from sklearn import utils
import sklearn
import pandas as pd


def normalize_data(dataset: np.ndarray, scaler: sklearn.preprocessing._data.StandardScaler) -> np.ndarray:
    dataset_scaled = scaler.fit_transform(dataset)
    dataset_scaled = np.expand_dims(dataset_scaled, axis=2)
    return dataset_scaled


def get_class_weights(y_train: pd.Series) -> dict:
    class_weights = utils.class_weight.compute_class_weight(
        'balanced', np.unique(y_train), y_train)
    class_weights = dict(enumerate(class_weights))
    return class_weights
