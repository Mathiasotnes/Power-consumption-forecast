import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split


def import_data(filepath, features):
    return pd.read_csv(filepath)[features]

def create_sequences(data, sequence_length):
    Xs = []
    for i in range(len(data) - sequence_length):
        Xs.append(data[i:(i + sequence_length)])
    return np.array(Xs)

def build_model(input_shape):
    return model

def plot_sequence(X, y, model):
    plt.figure(figsize=(12, 6))
    plt.plot(X, y, label='Actual')
    plt.plot(X, model.predict(X), label='Predicted')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    data_path = './data/processed_train.csv'
    features = ['NO1_consumption', 'NO1_temperature', 'time_of_day', 'time_of_week', 'time_of_year', 'NO1_consumption_lag_24', 'NO1_temperature_lag_24', 'NO1_consumption_mean_24', 'NO1_temperature_mean_24']
    data = import_data(data_path, features)

    # Create sequences
    sequence_length = 24
    X = create_sequences(data.values, sequence_length)
    y = data['NO1_consumption'].values[sequence_length:]

    # Build model
    input_shape = (X.shape[1], X.shape[2])
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape),
        LSTM(32, activation='tanh'),
        Dense(1, activation='relu')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Fit model
    history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, shuffle=False)

    # Save model
    model.save('./models/lstm_model.keras')
    pickle.dump(history.history, open('./models/lstm_history.pkl', 'wb'))

    # Plot sequence
    plot_sequence(X[-1], y[-1], model)
