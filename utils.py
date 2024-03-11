from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

MAX_EPOCHS = 20

features = ['NO1_consumption', 
            'NO1_temperature', 
            'time_of_day', 
            'time_of_week', 
            'time_of_year', 
            'NO1_consumption_lag_24', 
            'NO1_temperature_lag_24', 
            'NO1_consumption_mean_24', 
            'NO1_temperature_mean_24']

train_df = pd.read_csv('./data/train.csv')[features]
val_df = pd.read_csv('./data/val.csv')[features]
test_df = pd.read_csv('./data/test.csv')[features]

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
            self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def plot(self, model=None, plot_col='NO1_consumption', max_subplots=3, denormalize=False):
        scalar = joblib.load('./models/scaler.pkl')
        inputs, labels = self.example
        model_inputs = inputs
        inputs = inputs.numpy()
        labels = labels.numpy()

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]

        if denormalize:
            mean = scalar['mean'][plot_col_index]
            std = scalar['std'][plot_col_index]
            for i in range(inputs.shape[0]):
                inputs[i, :, plot_col_index] = inputs[i, :, plot_col_index] * std + mean
                labels[i, :, plot_col_index] = labels[i, :, plot_col_index] * std + mean

        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(model_inputs).numpy()

                if denormalize:
                    for i in range(predictions.shape[0]):
                        predictions[i, :, plot_col_index] = predictions[i, :, plot_col_index] * std + mean

                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)
        
        ds = ds.map(self.split_window)
        self.ds = ds

        return ds

    @staticmethod
    def compile_and_fit(model, window, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='min')

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.legacy.Adam(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    @test.setter
    def test(self, value):
        self._test = value

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
    
    @example.setter
    def example(self, value):
        """Set a new example batch of `inputs, labels`."""
        self._example = value
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])