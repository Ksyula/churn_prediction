import tensorflow as tf


class cnn1d_model():
    def __init__(self, input_shape: tuple, filters: int = 32, kernel_size: int = 10,
                 pool_size: int = 3, dense_nodes: int = 32, output_nodes: int = 1, output_activation: str = 'sigmoid'):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dense_nodes = dense_nodes
        self.output_nodes = output_nodes
        self.output_activation = output_activation

    def cnn1d_model(self) -> tf.keras.Model:
        """Generates a tf model

        Parameters
        ----------
        input_shape: tuple
            the shape of the input
        filters: int, default 32
            the number of filters in the Conv1D layers
        kernel_size: int, default 10
        pool_size: int, default 3
        dense_nodes: int, default 32
            the number of nodes in the Dense layer before the last layer
        output_node: int, default 1
            the number of node in the output layer
        output_activation: str, default 'sigmoid'
            the activation function in the outpur layer

        Returns
        -------
        tf.keras.Model
            the tf model
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.input_shape),
            tf.keras.layers.Conv1D(self.filters, self.kernel_size, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(self.pool_size),
            tf.keras.layers.Conv1D(self.filters, self.kernel_size, activation='relu', padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.dense_nodes, activation='relu'),
            tf.keras.layers.Dense(self.output_nodes, activation=self.output_activation)
        ])
        return model