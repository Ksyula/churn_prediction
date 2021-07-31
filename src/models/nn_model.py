import os.path as op
from datetime import datetime

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from sklearn import metrics

root_path = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
now = datetime.now().strftime("%d_%m_%Y_%H_%M")


def cnn1d_model(
    input_shape: tuple,
    filters: int = 32,
    kernel_size: int = 10,
    pool_size: int = 3,
    dense_nodes: int = 32,
    output_nodes: int = 1,
    output_activation: str = "sigmoid",
) -> tf.keras.Model:
    """
    Generates a tf model
    :param input_shape: tuple
        the shape of the input
    :param filters: int, default 32
        the number of filters in the Conv1D layers
    :param kernel_size: int, default 10
    :param pool_size: int, default 3
    :param dense_nodes: int, default 32
        the number of nodes in the Dense layer before the last layer
    :param output_nodes: int, default 1
        the number of node in the output layer
    :param output_activation: str, default 'sigmoid'
        the activation function in the outpur layer

    :return: tf.keras.Model
        the tf model
    """

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv1D(
                filters, kernel_size, activation="relu", padding="same"
            ),
            tf.keras.layers.MaxPooling1D(pool_size),
            tf.keras.layers.Conv1D(
                filters, kernel_size, activation="relu", padding="same"
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dense_nodes, activation="relu"),
            tf.keras.layers.Dense(output_nodes, activation=output_activation),
        ]
    )
    return model


def get_optimal_lr(X_train, y_train, class_weights, figure_path):

    model = cnn1d_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        filters=32,
        kernel_size=3,
        pool_size=2,
        dense_nodes=32,
    )
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-6 * 10 ** (epoch / 20)
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(lr=1e-6),
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        callbacks=[lr_schedule],
        class_weight=class_weights,
    )
    _plot_loss_per_lr(history, figure_path)
    optimal_lr = history.history["lr"][np.argmin(history.history["loss"])]
    return optimal_lr


def get_optimal_hps(
    X_train, y_train, X_val, y_val, optimal_lr, class_weights, model_path, seed
):
    def model_builder(hp):
        """
        Defines the model builder for keras tuner
        """
        hp_filters = hp.Int("filters", min_value=16, max_value=64, step=16)
        hp_kernel_size = hp.Int("kernel_size", min_value=3, max_value=5, step=1)
        hp_units = hp.Int("units", min_value=16, max_value=64, step=16)

        cnn_model = cnn1d_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            filters=hp_filters,
            kernel_size=hp_kernel_size,
            pool_size=2,
            dense_nodes=hp_units,
        )
        cnn_model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(lr=optimal_lr),
            metrics=["accuracy"],
        )
        return cnn_model

    # build tuner
    tuner = kt.BayesianOptimization(
        model_builder,
        objective="val_loss",
        max_trials=20,
        seed=seed,
        directory=_get_logs_path(model_path),
        project_name="tuning_kt_bayes",
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="auto",
        restore_best_weights=True,
    )

    # searching
    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        callbacks=[early_stopping],
        class_weight=class_weights,
        verbose=1,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps


def _plot_loss_per_lr(history, figure_path):
    """
    plot learning_rate-loss evolution and save it
    :param history:
    :return:
    """
    fig, ax = pyplot.subplots(figsize=(10, 6))
    ax.semilogx(history.history["lr"], history.history["loss"])
    ax.set_xlabel("learning rate")
    ax.set_ylabel("loss")
    ax.set_title("Loss evolution along with learning rates")
    fig.tight_layout()
    fig.savefig(figure_path, format="png")


def _get_logs_path(model_path):
    log_name = "logs_" + now
    model_path = op.join(root_path, model_path, log_name)
    return model_path


def evaluate_model(
    opt_cnn_model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    best_lr,
    best_hps,
    evaluation_report_path,
):
    y_pred_train = opt_cnn_model.predict(X_train)
    y_pred_train = y_pred_train > 0.5
    train = metrics.classification_report(y_train, y_pred_train)

    y_pred_val = opt_cnn_model.predict(X_val)
    y_pred_val = y_pred_val > 0.5
    valid = metrics.classification_report(y_val, y_pred_val)

    y_pred_test = opt_cnn_model.predict(X_test)
    y_pred_test = y_pred_test > 0.5
    test = metrics.classification_report(y_test, y_pred_test)

    with open(evaluation_report_path, "w") as evaluation_report:
        print("\nThe best hyperparameters are:", file=evaluation_report)
        print(f"\tLearning_rate: {best_lr}", file=evaluation_report)
        print(f"\tFilter: {best_hps.get('filters')}", file=evaluation_report)
        print(f"\tKernel_size: {best_hps.get('kernel_size')}", file=evaluation_report)
        print(f"\tUnits: {best_hps.get('units')}", file=evaluation_report)
        print(
            f"Training set:\n {train}\n\n Validation set:\n {valid}\n\n Test set:\n {test}\n\n",
            file=evaluation_report,
        )
