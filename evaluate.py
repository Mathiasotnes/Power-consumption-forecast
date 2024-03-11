import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

def plot_training_progress(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Add accuracy plotting if your model uses accuracy as a metric.
    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['accuracy'], label='Train Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # plt.title('Accuracy Progress')
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

def import_data(filepath, features):
    return pd.read_csv(filepath)[features]

def create_sequences(data, sequence_length):
    Xs = []
    for i in range(len(data) - sequence_length):
        Xs.append(data[i:(i + sequence_length)])
    return np.array(Xs)

def plot_observed_vs_forecasted(input_data, actual_future, predicted_future):
    plt.figure(figsize=(15, 6))
    
    # Create a time axis for the plot, 48 hours in total
    time_axis = range(-24, 24)
    
    # Split the actual data into the past (input) and the future (actual outcome)
    actual_input = input_data[-48:-24]  # Last 24 hours of the input
    actual_future = actual_future[-24:]  # Actual future values
    
    # The predicted future is the forecast for the next 24 hours
    predicted_future = predicted_future[-24:]
    
    # Plot the actual input data
    plt.plot(time_axis[:24], actual_input, label='Input', color='blue')
    
    # Plot the actual future data
    plt.plot(time_axis[24:], actual_future, label='Actual Future', color='green')
    
    # Plot the predicted future data
    plt.plot(time_axis[24:], predicted_future, label='Forecasted', color='red', linestyle='--')
    
    plt.axvline(x=0, color='black', linestyle='--')  # Divider between past and future
    plt.title('Input, Observed, vs Forecasted Consumption')
    plt.xlabel('Time (hours)')
    plt.ylabel('Consumption')
    plt.legend()
    plt.show()

def plot_error_distribution(errors):
    mean_errors = np.mean(errors, axis=0)
    std_errors = np.std(errors, axis=0)
    hours = range(1, 25)

    plt.figure(figsize=(10, 6))
    plt.errorbar(hours, mean_errors, yerr=std_errors, fmt='-o')
    plt.title('Error Distribution Over Forecast Horizon')
    plt.xlabel('Forecast Horizon (hours)')
    plt.ylabel('Absolute Error')
    plt.show()

def plot_history(history, val=False):
    plt.figure(figsize=(12, 6))
    plt.plot(history['loss'], label='loss')
    if val:
        plt.plot(history['val_loss'], label='val_loss')
    plt.title('Model Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_model_comparison(metrics, model_names):
    df = pd.DataFrame(metrics, index=model_names)
    df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Comparison')
    plt.xlabel('Model')
    plt.ylabel('Metric Value')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    model_path = './models/lstm_model.keras'
    model = load_model(model_path)
    data_path = './data/processed_test.csv'
    features = ['NO1_consumption', 'NO1_temperature', 'time_of_day', 'time_of_week', 'time_of_year', 'NO1_consumption_lag_24', 'NO1_temperature_lag_24', 'NO1_consumption_mean_24', 'NO1_temperature_mean_24']
    data = import_data(data_path, features)

    # plot_training_progress(model.history)

    
    ############## LSTM Model ##############

    # Plot predictions versus actual values
    sequence_length = 24
    test_data = data.values
    X_test = create_sequences(test_data, sequence_length)
    y_test = data['NO1_consumption'].values[sequence_length:]

    predicted = model.predict(X_test) 
    actual = y_test

    plot_observed_vs_forecasted(actual[-48:], actual[-24:], predicted[-24:])

    # Training history
    history = pickle.load(open('./models/lstm_history.pkl', 'rb'))
    plot_history(history)

    
    # errors = actual - predicted
    # plot_error_distribution(errors)

    # For model comparison, assume you have metrics stored for each model
    # metrics = {'RMSE': [value1, value2], 'MAE': [value1, value2]}
    # model_names = ['Model 1', 'Model 2']
    # plot_model_comparison(metrics, model_names)
