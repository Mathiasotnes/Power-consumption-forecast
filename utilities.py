import numpy as np

import numpy as np

def n_in_n_out(data, n_in, n_out, target_column):
    """
    Prepare data for "n_in_n_out" predictions, compatible with TensorFlow's LSTM.
    
    Parameters:
    - data: numpy array of shape (samples, features), including the target column.
    - n_in: Number of input time steps.
    - n_out: Number of output time steps to predict.
    - target_column: Index of the target column in `data`.
    
    Returns:
    - X: Input data for the LSTM, with shape (samples, n_in, features).
    - y: Target data for the LSTM, with shape (samples, n_out).
    """
    X, y = [], []
    for i in range(len(data) - n_in - n_out + 1):
        # Select input sequence and the corresponding future output sequence
        X.append(data[i:(i + n_in), :])
        y.append(data[(i + n_in):(i + n_in + n_out), target_column])
    
    return np.array(X), np.array(y)


def n_in_1_out(data, n_in, target_column):
    """
    Prepare data for "n_in_1_out" predictions, compatible with TensorFlow's LSTM.
    
    Parameters:
    - data: numpy array of shape (samples, features), including the target column.
    - n_in: Number of input time steps.
    - target_column: Index of the target column in `data`.
    
    Returns:
    - X: Input data for the LSTM, with shape (samples, n_in, features).
    - y: Target data for the LSTM, with shape (samples, 1).

    Timesteps:
    X: [t-n_in, t-n_in+1, ..., t-1]
    y: [t]
    """
    X, y = [], []
    for i in range(len(data) - n_in):
        # Select input sequence and the corresponding future output
        X.append(data[i:(i + n_in), :])
        y.append(data[i + n_in, target_column])
    
    return np.array(X), np.array(y)
