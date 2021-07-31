import logging
import os.path as op

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection, preprocessing

from config import Config
from src.models.nn_model import (
    cnn1d_model,
    evaluate_model,
    get_optimal_hps,
    get_optimal_lr,
)
from src.models.utils import get_class_weights, normalize_data

root_path = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))


def main(config: dict):
    """Run model training on built featureset (from data/processed)"""
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Reed featureset
    featureset = pd.read_csv(op.join(root_path, config["data_processed_path"]))

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

    ## Define model and tune hyperparams
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)
    seed = 10
    figure_path = op.join(root_path, config["figures_path"])

    # 1. Learning rate
    best_lr = get_optimal_lr(X_train_scaled, y_train, class_weights, figure_path)
    # best_lr = 0.014125375
    # 2. Filters, kernel_size, units via keras tuner
    best_hps = get_optimal_hps(X_train_scaled, y_train, X_val_scaled, y_val, best_lr, class_weights, config["model_path"], seed)
    # best_hps = {"filters": 16, "kernel_size": 3, "units": 16}

    # training model with the optimal hps
    tf.keras.backend.clear_session()
    opt_cnn_model = cnn1d_model(
        input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
        filters=best_hps.get("filters"),
        kernel_size=best_hps.get("kernel_size"),
        pool_size=2,
        dense_nodes=best_hps.get("units"),
    )
    opt_cnn_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(lr=best_lr),
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="auto",
        restore_best_weights=True,
    )
    opt_cnn_model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=500,
        callbacks=[early_stopping],
        class_weight=class_weights,
    )

    evaluation_report_path = op.join(root_path, config["evaluation_report"])
    evaluate_model(
        opt_cnn_model,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_test_scaled,
        y_test,
        best_lr,
        best_hps,
        evaluation_report_path,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    config = Config()
    main(config.input_dict)
