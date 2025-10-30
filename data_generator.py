import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_workload_data(num_servers=3, num_timesteps=1000, time_interval_minutes=5):
    """
    Generate synthetic time-series data for server workloads (CPU usage).

    Parameters:
    - num_servers: Number of servers to simulate
    - num_timesteps: Number of time steps to generate
    - time_interval_minutes: Minutes between each time step

    Returns:
    - DataFrame with columns: timestamp, server_id, cpu_usage
    """
    np.random.seed(42)  # For reproducibility

    data = []
    start_time = datetime.now()

    for server_id in range(num_servers):
        # Base CPU usage with trend and seasonality
        base_usage = 50 + 10 * np.sin(2 * np.pi * np.arange(num_timesteps) / 144)  # Daily cycle (144 * 5min = 12 hours)
        trend = 0.01 * np.arange(num_timesteps)  # Slight upward trend
        noise = np.random.normal(0, 5, num_timesteps)  # Random noise

        cpu_usage = base_usage + trend + noise
        cpu_usage = np.clip(cpu_usage, 0, 100)  # Clip to 0-100%

        for t in range(num_timesteps):
            timestamp = start_time + timedelta(minutes=t * time_interval_minutes)
            data.append({
                'timestamp': timestamp,
                'server_id': f'server_{server_id}',
                'cpu_usage': cpu_usage[t]
            })

    df = pd.DataFrame(data)
    return df

def prepare_lstm_data(df, lookback=10):
    """
    Prepare data for LSTM training by creating sequences.

    Parameters:
    - df: DataFrame with workload data
    - lookback: Number of previous time steps to use for prediction

    Returns:
    - X: Input sequences
    - y: Target values
    """
    X, y = [], []
    for server_id in df['server_id'].unique():
        server_data = df[df['server_id'] == server_id]['cpu_usage'].values
        for i in range(len(server_data) - lookback):
            X.append(server_data[i:i+lookback])
            y.append(server_data[i+lookback])

    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)

    return X, y

if __name__ == "__main__":
    # Example usage
    df = generate_synthetic_workload_data()
    print(df.head())
    print(f"Data shape: {df.shape}")

    X, y = prepare_lstm_data(df)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
