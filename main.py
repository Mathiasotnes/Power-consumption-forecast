import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import torch
import torch.optim as optim
import torch.utils.data as data

def create_dataset(dataset, lookback, target_column_index, mode='lstm'):
    """
    Creates datasets for the training where one sample has the timestep indexes:
    X = [0, 1, 2, 3, ..., lookback-1]
    y = [lookback]
    """
    X, y = [], []
    dataset = dataset.to_numpy()

    if mode == 'lstm' or mode == 'transformer':
        for i in range(len(dataset)-lookback-1):
            feature = dataset[i:i+lookback, :]
            target = dataset[i+lookback, target_column_index]
            X.append(feature)
            y.append(target)
        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float).unsqueeze(1) 
    
    elif mode == 'cnn':
        # Reshape for CNN (batch, features, seq_len)
        for i in range(len(dataset)-lookback-1):
            feature = dataset[i:i+lookback, :].T  # Transpose to get (features, seq_len)
            target = dataset[i+lookback, target_column_index]
            X.append(feature)
            y.append(target)

    return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float).unsqueeze(1)
        

def predict_future(model, initial_sequence, steps=24, input_steps=24, mode='lstm'):
    model.eval()
    current_sequence = initial_sequence.clone().detach()
    predictions = []

    if mode == 'lstm' or mode == 'transformer':
        with torch.no_grad():
            for i in range(steps):
                current_input = current_sequence[i:i+input_steps, :].view(1, input_steps, -1)
                next_value = model(current_input)
                if mode == 'transformer':
                    next_value = torch.mean(next_value, dim=0)
                predictions.append(next_value.item())

                # Overwrite the current_sequence with the predicted value
                current_sequence[i+input_steps, 0] = next_value
            
    elif mode == 'cnn':
        start_index = 17
        with torch.no_grad():
            for i in range(steps):
                current_input = current_sequence[start_index+i:start_index+input_steps+i, :].T.unsqueeze(0)
                next_value = model(current_input)
                predictions.append(next_value.item())

                # Overwrite the current_sequence with the predicted value
                current_sequence[start_index+input_steps+i+1, 0] = next_value

    return predictions

def predict_future_full_dataset(model, dataset, lookback, target_column_index, steps=24, mode='lstm'):
    """
    Produces 24 hour auto-regressive predictions for the entire dataset at every time step.
    """
    model.eval()
    num_predictions = len(dataset) - 2 * lookback
    predictions = np.zeros((num_predictions, steps))
    labels = np.zeros((num_predictions, steps))

    if mode == 'lstm' or mode == 'transformer':
        with torch.no_grad():
            for i in range(num_predictions):
                first_day = dataset[i, :, :]
                second_day = dataset[i + lookback, :, :]
                current_sequence = torch.cat((first_day, second_day), dim=0)
                current_labels = dataset[i + lookback:i + lookback + steps, 0, target_column_index]
                current_prediction = predict_future(model, current_sequence, steps=steps, mode=mode)
                
                predictions[i, :] = current_prediction
                labels[i, :] = current_labels.numpy()
    
    elif mode == 'cnn':
        with torch.no_grad():
            for i in range(num_predictions-48):
                day_one = dataset[i:i+24:lookback, :].transpose(1, 2).flatten(start_dim=0, end_dim=1)
                day_two = dataset[i+24:i+48:lookback, :].transpose(1, 2).flatten(start_dim=0, end_dim=1)
                current_sequence = torch.cat((day_one, day_two), dim=0)
                current_labels = day_two[0:24, target_column_index]
                current_prediction = predict_future(model, current_sequence, steps=steps, input_steps=lookback, mode=mode)

                if day_one.shape[0] != 24 or day_two.shape[0] != 24:
                    break

                predictions[i, :] = current_prediction
                labels[i, :] = current_labels.numpy()

    return predictions, labels

def train_model(model, X_train, y_train, X_test, y_test, n_epochs=15, save=False, save_model_path=None, save_history_path=None):
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=24)
    history = {'train': {'mse': [], 'mae': []}, 'val': {'mse': [], 'mae': []}}

    for epoch in range(n_epochs):
        model.train()
        train_mse_accum = 0.0
        train_mae_accum = 0.0
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_mse_accum += loss_fn(y_pred, y_batch).item() * X_batch.size(0)
            train_mae_accum += torch.abs(y_pred - y_batch).sum().item()

        # Normalize the accumulated loss by the total number of samples
        train_mse = train_mse_accum / len(X_train)
        train_mae = train_mae_accum / len(X_train)

        history['train']['mse'].append(train_mse)
        history['train']['mae'].append(train_mae)

        # Validation
        model.eval()
        val_mse_accum = 0.0
        val_mae_accum = 0.0
        with torch.no_grad():
            y_pred = model(X_test)
            val_mse_accum += loss_fn(y_pred, y_test).item() * X_test.size(0)
            val_mae_accum += torch.abs(y_pred - y_test).sum().item()

        # Normalize the accumulated loss by the total number of samples
        val_mse = val_mse_accum / len(X_test)
        val_mae = val_mae_accum / len(X_test)

        history['val']['mse'].append(val_mse)
        history['val']['mae'].append(val_mae)

        print(f"Epoch {epoch} | train MSE {train_mse:.4f} | val MSE {val_mse:.4f} | train MAE: {train_mae:.4f} | val MAE {val_mae:.4f}")

    if save:
        torch.save(model.state_dict(), save_model_path)
        pickle.dump(history, open(save_history_path, 'wb'))

    return model, history


def plot_predictions(day_one_values, predictions, true_values, day_one_label='Day 1', prediction_label='Predictions', true_values_label='True values', title='Prediction vs True Values', denormalize=False, save=False, save_path=None):
    plt.figure(figsize=(10, 6))

    if denormalize:
        day_one_values = denormalize_data(day_one_values)
        predictions = denormalize_data(predictions)
        true_values = denormalize_data(true_values)
    
    # Day one values
    plt.plot(range(len(day_one_values)), day_one_values, label=day_one_label, color='blue', linewidth=2)
    
    # Predictions
    plt.plot(range(len(day_one_values), len(day_one_values) + len(predictions)), predictions, 'x', label=prediction_label, color='orange', linewidth=2)
    
    # True values
    plt.plot(range(len(day_one_values), len(day_one_values) + len(true_values)), true_values, 'o', label=true_values_label, color='green', linewidth=2)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(save_path)
    
    plt.show()

def denormalize_data(data):
    denormalized = []
    for value in data:
        denormalized.append(value * scaler_std + scaler_mean)
    return denormalized

def plot_history(history, save=False, save_path=None):
    # Create subplots for mae and mse
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Plot mse
    ax[0].plot(history['train']['mse'], label='train mse')
    ax[0].plot(history['val']['mse'], label='val mse')
    ax[0].set_title('MSE')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('MSE')
    ax[0].legend()

    # Plot mae
    ax[1].plot(history['train']['mae'], label='train mae')
    ax[1].plot(history['val']['mae'], label='val mae')
    ax[1].set_title('MAE')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('MAE')
    ax[1].legend()

    if save:
        plt.savefig(save_path)

    plt.show()

def plot_error(predictions, true_values, save=False, save_path=None):
    # predictions and true_values are arrays of shape (n_samples, n_steps)
    
    # First, ensure all arrays are of equal length (n_steps)
    n_steps = predictions[0].shape[0]

    # Initialize an array to accumulate the errors for each step
    accumulated_errors = np.zeros(n_steps)

    # Loop over each prediction and true value pair
    for pred, true in zip(predictions, true_values):
        # Compute the error for this sample
        error = pred - true
        # Square the errors for variance computation
        absolute_error = np.abs(error)
        # Accumulate the squared errors
        accumulated_errors += absolute_error

    # Divide by the number of samples to get the variance, then take the square root to get std
    std_errors = np.sqrt(accumulated_errors / len(predictions))
    
    plt.figure(figsize=(10, 6))
    hours = np.arange(1, 25)
    plt.bar(hours, std_errors)
    plt.xlabel('Hour')
    plt.ylabel('Standard Deviation of Errors')
    plt.title('Standard Deviation of Prediction Errors by Hour')
    plt.xticks(hours) 
    plt.grid(True)

    if save:
        plt.savefig(save_path)

    plt.show()

def plot_errors(predictions_array, true_values, save=False, save_path=None):
        n_steps = predictions_array[0][0].shape[0]
        accumulated_errors_array = np.zeros((len(predictions_array), n_steps))
        for i, predictions in enumerate(predictions_array):
            for pred, true in zip(predictions, true_values):
                error = pred - true
                absolute_error = np.abs(error)
                accumulated_errors_array[i, :] += absolute_error
        std_errors_array = np.sqrt(accumulated_errors_array / len(predictions_array))
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        hours = np.arange(1, 25)
        n_bars = len(std_errors_array)
        bar_width = 0.75 / n_bars
        
        # Offset calculation to position bars side by side
        offsets = np.linspace(-bar_width*n_bars/3, bar_width*n_bars/3, n_bars)
        labels = ["Transformer", "CNN", "LSTM trained on N01", "LSTM trained on N05"]
        
        for i, std_errors in enumerate(std_errors_array):
            plt.bar(hours + offsets[i], std_errors, width=bar_width, label=labels[i])
        plt.xlabel('Hour')
        plt.ylabel('MAE')
        plt.title('Mean absolute errors of Predictions on N05, by hours into the future')
        plt.xticks(hours)
        plt.legend()
        if save:
            plt.savefig(save_path)
        plt.show()
    
def plot_objective(X_train, y_train, target_column_index):
    X_train = denormalize_data(X_train.numpy()[0, :, target_column_index])
    y_train = denormalize_data(y_train.numpy()[0, :])
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(X_train)), X_train, 'x', label='Inputs')
    plt.plot(range(len(X_train), len(X_train) + len(y_train)), y_train, 'o', label='Target', )
    plt.title('Objective: Predict target value from input values')
    plt.xlabel('Timestep')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()
 
class LstmModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=8, hidden_size=50, num_layers=2, batch_first=True)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.linear = torch.nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        self.batch_norm = torch.nn.BatchNorm1d(num_features=100)
        x = self.linear(x)
        return x
    
class CnnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1d parameters: in_channels, out_channels, kernel_size
        self.conv1 = torch.nn.Conv1d(in_channels=8, out_channels=64, kernel_size=6, padding='same')
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.2)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=6, padding='same')
        self.relu2 = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x):
        # PyTorch Conv1d expects input shape of (batch, channels, seq_len)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

class TransformerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embedding = torch.nn.Linear(8, 24)
        self.pos_encoder = PositionalEncoding(24, dropout=0.2)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model=24, nhead=8, dim_feedforward=64, dropout=0.2, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers=6)
        self.decoder = torch.nn.Linear(24, 1)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.input_embedding(x)  # Convert to (batch_size, seq_len, 512)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Taking the last time step
        x = self.decoder(x)
        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        try:
            x = x + self.pe[:x.size(1)]
        except:
            try:
                x = x + self.pe[:x.size(1), :]
            except:
                return self.dropout(x)
        return self.dropout(x)


if __name__ == '__main__':

    train_data_path = './data/train.csv'
    test_data_path = './data/test.csv'
    val_data_path = './data/val.csv'

    features = ['NO5_consumption',
                'NO5_temperature',
                'time_of_day',
                'time_of_week',
                'time_of_year',
                'NO5_consumption_lag_24',
                'NO5_temperature_lag_24',
                'NO5_temperature_mean_24']
    
    target_column = 'NO5_consumption'

    scaler = pickle.load(open('./models/scaler.pkl', 'rb'))
    scaler_mean = scaler['mean'][target_column]
    scaler_std = scaler['std'][target_column]

    train = pd.read_csv(train_data_path)[features]
    test = pd.read_csv(test_data_path)[features]
    val = pd.read_csv(val_data_path)[features]
    target_column_index = train.columns.get_loc(target_column)

    # LSTM mode size        => (batch, seq_len, features)
    # CNN mode size         => (batch, features, seq_len)
    # Transformer mode size => (batch, seq_len, features)
    lookback = 24
    X_train, y_train = create_dataset(train, lookback=lookback, target_column_index=target_column_index, mode='transformer')
    X_test, y_test = create_dataset(test, lookback=lookback, target_column_index=target_column_index, mode='transformer')
    X_val, y_val = create_dataset(val, lookback=lookback, target_column_index=target_column_index, mode='transformer')
    # plot_objective(X_train, y_train, target_column_index)

    # Train and save the model

    # LSTM
    # model = LstmModel()
    # model, history = train_model(model, X_train, y_train, X_val, y_val, n_epochs=50, save=True, save_model_path='./models/so_lstm_n05.pth', save_history_path='./models/so_lstm_n05_history.pkl')
    # plot_history(history)

    # CNN
    # model = CnnModel()
    # model, history = train_model(model, X_train, y_train, X_val, y_val, n_epochs=50, save=True, save_model_path='./models/so_cnn_n05.pth', save_history_path='./models/so_cnn_n05_history.pkl')
    # plot_history(history)

    # Transformer
    # model = TransformerModel()
    # model, history = train_model(model, X_train, y_train, X_val, y_val, n_epochs=30, save=True, save_model_path='./models/transformer_n05.pth', save_history_path='./models/transformer_n05_history.pkl')
    # plot_history(history)

    # Load the model
    model = TransformerModel()
    state_dict = torch.load('./models/transformer_n05.pth')
    history = pickle.load(open('./models/transformer_n05_history.pkl', 'rb'))
    model.load_state_dict(state_dict=state_dict, strict=False)

    # Plot the training history
    # plot_history(history, save=True, save_path='./figures/cnn_n05_history.png')

    # Predict the next 24 hours

    # LSTM and Transformer
    day_one = X_test[0]
    day_two = X_test[24]
    initial_sequence = torch.cat((day_one, day_two), dim=0)

    # CNN
    # day_one = X_test[0:24:lookback, :].transpose(1, 2).flatten(start_dim=0, end_dim=1)
    # day_two = X_test[24:48:lookback, :].transpose(1, 2).flatten(start_dim=0, end_dim=1)
    # initial_sequence = torch.cat((day_one, day_two), dim=0)
    # day_one = day_one.numpy()
    # day_two = day_two.numpy()

    # predictions = predict_future(model, initial_sequence, steps=24, input_steps=lookback, mode='transformer')
    # y_test = y_test.numpy()

    # plot_predictions(day_one[:, 0], predictions, day_two[:, 0], save=True, save_path='./figures/transformer_n05_pred_n05.png', denormalize=True)
    
    # Predict the entire test set
    # predictions, labels = predict_future_full_dataset(model, X_test, lookback, target_column_index, steps=24, mode='transformer')
    # predictions = denormalize_data(predictions)
    # labels = denormalize_data(labels)
    # predictions = (predictions, labels)

    # Save the predictions
    # with open('./models/transformer_n05_pred_n05.pkl', 'wb') as f:
    #     pickle.dump(predictions, f)

    # Load the predictions
    with open('./models/transformer_n05_pred_n05.pkl', 'rb') as f:
        predictions_transformer = pickle.load(f)

    with open('./models/cnn_n05_pred_n05.pkl', 'rb') as f:
        predictions_cnn = pickle.load(f)

    with open('./models/so_lstm_n01_pred_n05.pkl', 'rb') as f:
        predictions_lstm_n01 = pickle.load(f)

    with open('./models/so_lstm_n05_pred_n05.pkl', 'rb') as f:
        predictions_lstm_n05 = pickle.load(f)
    


    # Plot the predictions
    # plot_predictions(predictions[1][0], predictions[0][24], predictions[1][24])
    # plot_predictions(predictions[1][24], predictions[0][48], predictions[1][48])
    # plot_predictions(predictions[1][48], predictions[0][72], predictions[1][72])
    # plot_predictions(predictions[1][72], predictions[0][96], predictions[1][96])
    # plot_error(predictions[0], predictions[1], save=True, save_path='./figures/transformer_n05_error_n05.png')
    plot_errors([predictions_transformer[0], predictions_cnn[0], predictions_lstm_n01[0], predictions_lstm_n05[0]], predictions_transformer[1], save=True, save_path='./figures/all_models_error_n05.png')

    
