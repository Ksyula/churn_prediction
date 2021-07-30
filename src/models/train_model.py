import logging

import numpy as np
import pandas as pd
from config import Config
from sklearn import metrics, model_selection, preprocessing, utils

from src.models.utils import get_class_weights, normalize_data


def main(config: dict):
    """Run model training on built featureset (from data/processed)"""
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Reed featureset
    featureset = pd.read_csv(config["data_processed_path"])

    # Prepare featureset
    featureset.set_index("customerID", inplace=True)
    target = featureset.pop("Churn")
    feature_matrix = featureset.values

    # Train/validation/test split
    X_train_val, X_test, y_train_val, y_test = model_selection.train_test_split(
        feature_matrix,
        target,
        train_size=config["train_size"],
        random_state=config["random_state"],
    )
    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X_train_val,
        y_train_val,
        train_size=config["train_size"],
        random_state=config["random_state"],
    )

    # Normalization
    scaler = preprocessing.StandardScaler()
    X_train_scaled = normalize_data(X_train, scaler)
    X_val_scaled = normalize_data(X_val, scaler)
    X_test_scaled = normalize_data(X_test, scaler)

    # Class weights
    class_weights = get_class_weights(y_train)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    config = Config()
    main(config)
