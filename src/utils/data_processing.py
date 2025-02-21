import numpy as np

def create_sequences(data, look_back=168, forecast_horizon=24):

    X, y = [], []

    for i in range(len(data) - look_back - forecast_horizon):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back:i + look_back + forecast_horizon])

    return np.array(X), np.array(y)