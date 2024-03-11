import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(input_file):
    return pd.read_csv(input_file, parse_dates=['timestamp'], index_col='timestamp')

def add_time_features(df):
    df['time_of_day'] = df.index.hour
    df['time_of_week'] = df.index.dayofweek
    df['time_of_year'] = df.index.dayofyear

def add_time_sine_cosine_features(df):
    df['sin_time_of_day'] = np.sin(2 * np.pi * df.index.hour / 23.0)
    df['cos_time_of_day'] = np.cos(2 * np.pi * df.index.hour / 23.0)
    df['sin_time_of_week'] = np.sin(2 * np.pi * df.index.dayofweek / 6.0)
    df['cos_time_of_week'] = np.cos(2 * np.pi * df.index.dayofweek / 6.0)
    df['sin_time_of_year'] = np.sin(2 * np.pi * df.index.dayofyear / 365.0)
    df['cos_time_of_year'] = np.cos(2 * np.pi * df.index.dayofyear / 365.0)

def add_lag_features(df, lag_hours=24):
    for i in range(1, 6):
        consumption_col = f'NO{i}_consumption'
        temperature_col = f'NO{i}_temperature'
        
        df[f'{consumption_col}_lag_{lag_hours}'] = df[consumption_col].shift(lag_hours)
        df[f'{temperature_col}_lag_{lag_hours}'] = df[temperature_col].shift(lag_hours)

def add_mean_features(df, window_size=24):
    for i in range(1, 6):
        consumption_col = f'NO{i}_consumption'
        temperature_col = f'NO{i}_temperature'
        
        df[f'{consumption_col}_mean_{window_size}'] = df[consumption_col].rolling(window=window_size).mean()
        df[f'{temperature_col}_mean_{window_size}'] = df[temperature_col].rolling(window=window_size).mean()

def add_timestep_feature(df):
    df['timestep'] = range(len(df))

def normalize_data(train_df, test_df, val_df):
    # Store column names
    columns = train_df.columns

    # Normalize the data
    scaler = StandardScaler()
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)
    val_df = scaler.transform(val_df)

    # Convert back to DataFrame
    train_df = pd.DataFrame(train_df, columns=columns)
    test_df = pd.DataFrame(test_df, columns=columns)
    val_df = pd.DataFrame(val_df, columns=columns)

    # Store the normalization
    joblib.dump(scaler, './models/scaler.gz')

    return train_df, test_df, val_df

def preprocess_data(input_file, features_to_add, split_data=True, test_size=0.15, val_size=0.15):
    df = load_data(input_file)
    
    if 'time_features' in features_to_add:
        add_time_features(df)
        
    if 'lag_features' in features_to_add:
        add_lag_features(df)

    if 'mean_features' in features_to_add:
        add_mean_features(df)

    if 'timestep_feature' in features_to_add:
        add_timestep_feature(df)
    
    if 'time_sine_cosine_features' in features_to_add:
        add_time_sine_cosine_features(df)

    df.dropna(inplace=True)

    if split_data:
        train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)
        train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), shuffle=False)

    if 'normalize' in features_to_add:
        train_df, test_df, val_df = normalize_data(train_df, test_df, val_df)

    train_df.to_csv('./data/train.csv')
    test_df.to_csv('./data/test.csv')
    val_df.to_csv('./data/val.csv')

if __name__ == '__main__':
    input_file = 'data/consumption_and_temperatures.csv'
    features_to_add = ['time_features', 'lag_features', 'mean_features', 'timestep_feature', 'time_sine_cosine_features', 'normalize']
    split_data = True
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15
    
    preprocess_data(input_file, features_to_add, split_data=split_data, test_size=test_size, val_size=val_size)
