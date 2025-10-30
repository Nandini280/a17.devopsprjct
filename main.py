import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_synthetic_workload_data, prepare_lstm_data
from lstm_model import LSTMWorkloadPredictor, create_server_predictors
from workload_balancer import SmartWorkloadBalancer, simulate_balancing

def main():
    """
    Main function to run the Smart DevOps Workload Balancer.
    """
    print("Starting Smart DevOps Workload Balancer...")

    # Step 1: Generate synthetic workload data
    print("Generating synthetic workload data...")
    df = generate_synthetic_workload_data(num_servers=3, num_timesteps=1000)
    print(f"Generated data shape: {df.shape}")
    print(df.head())

    # Step 2: Train LSTM predictors for each server
    print("Training LSTM predictors...")
    predictors = create_server_predictors(df, lookback=10)
    print(f"Trained predictors for {len(predictors)} servers")

    # Step 3: Simulate workload balancing
    print("Simulating workload balancing...")
    simulate_balancing(predictors, num_tasks=20)

    # Step 4: Visualize results (optional)
    print("Generating visualizations...")
    plot_workload_data(df)
    plot_predictions(df, predictors)

    print("Workload balancing simulation completed!")

def plot_workload_data(df):
    """
    Plot the synthetic workload data for all servers.
    """
    plt.figure(figsize=(12, 6))
    for server_id in df['server_id'].unique():
        server_data = df[df['server_id'] == server_id]
        plt.plot(server_data['timestamp'], server_data['cpu_usage'], label=server_id)

    plt.title('Synthetic Server Workload Data')
    plt.xlabel('Time')
    plt.ylabel('CPU Usage (%)')
    plt.legend()
    plt.savefig('workload_data.png')
    plt.show()

def plot_predictions(df, predictors):
    """
    Plot actual vs predicted CPU usage for a sample server.
    """
    server_id = 'server_0'
    server_data = df[df['server_id'] == server_id]['cpu_usage'].values
    predictor = predictors[server_id]

    # Prepare test data
    X, y = prepare_lstm_data(df[df['server_id'] == server_id])
    predictions = predictor.predict(X)

    plt.figure(figsize=(12, 6))
    plt.plot(y[:100], label='Actual', alpha=0.7)
    plt.plot(predictions[:100], label='Predicted', alpha=0.7)
    plt.title(f'Actual vs Predicted CPU Usage for {server_id}')
    plt.xlabel('Time Steps')
    plt.ylabel('CPU Usage (%)')
    plt.legend()
    plt.savefig('predictions.png')
    plt.show()

if __name__ == "__main__":
    main()
