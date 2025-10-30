import numpy as np
from typing import List, Dict, Tuple

class SmartWorkloadBalancer:
    def __init__(self, predictors: Dict[str, 'LSTMWorkloadPredictor'], num_servers: int):
        """
        Initialize the workload balancer.

        Parameters:
        - predictors: Dictionary of LSTM predictors for each server
        - num_servers: Number of servers in the system
        """
        self.predictors = predictors
        self.num_servers = num_servers
        self.server_loads = {f'server_{i}': [] for i in range(num_servers)}  # Historical loads
        self.task_queue = []  # Queue of tasks to be assigned

    def predict_future_loads(self, server_id: str, current_sequence: np.ndarray, steps_ahead: int = 1) -> float:
        """
        Predict future CPU load for a server.

        Parameters:
        - server_id: ID of the server
        - current_sequence: Recent CPU usage sequence
        - steps_ahead: Number of steps to predict ahead

        Returns:
        - Predicted CPU load
        """
        predictor = self.predictors[server_id]
        prediction_input = current_sequence.reshape(1, -1, 1)

        predictions = []
        for _ in range(steps_ahead):
            pred = predictor.predict(prediction_input)[0]
            predictions.append(pred)
            # Update input for next prediction (sliding window)
            prediction_input = np.roll(prediction_input, -1, axis=1)
            prediction_input[0, -1, 0] = pred

        return predictions[-1]  # Return the final prediction

    def assign_task(self, task_load: float, current_loads: Dict[str, float]) -> str:
        """
        Assign a task to the server with the lowest predicted load.

        Parameters:
        - task_load: Estimated CPU load of the task
        - current_loads: Current CPU loads of all servers

        Returns:
        - Server ID to assign the task to
        """
        best_server = None
        lowest_predicted_load = float('inf')

        for server_id in current_loads.keys():
            if server_id not in self.predictors:
                continue

            # Get recent sequence for prediction
            recent_loads = self.server_loads[server_id][-self.predictors[server_id].lookback:]
            if len(recent_loads) < self.predictors[server_id].lookback:
                # Not enough data, use current load
                predicted_load = current_loads[server_id] + task_load
            else:
                current_sequence = np.array(recent_loads)
                predicted_load = self.predict_future_loads(server_id, current_sequence) + task_load

            if predicted_load < lowest_predicted_load:
                lowest_predicted_load = predicted_load
                best_server = server_id

        return best_server

    def update_server_load(self, server_id: str, new_load: float):
        """
        Update the historical load data for a server.

        Parameters:
        - server_id: ID of the server
        - new_load: New CPU load measurement
        """
        self.server_loads[server_id].append(new_load)
        # Keep only recent history
        max_history = max(p.lookback for p in self.predictors.values()) * 2
        if len(self.server_loads[server_id]) > max_history:
            self.server_loads[server_id] = self.server_loads[server_id][-max_history:]

    def balance_workload(self, incoming_tasks: List[float], current_loads: Dict[str, float]) -> Dict[str, List[float]]:
        """
        Balance incoming tasks across servers using LSTM predictions.

        Parameters:
        - incoming_tasks: List of task loads to assign
        - current_loads: Current CPU loads of all servers

        Returns:
        - Dictionary mapping server IDs to lists of assigned task loads
        """
        assignments = {f'server_{i}': [] for i in range(self.num_servers)}

        for task_load in incoming_tasks:
            assigned_server = self.assign_task(task_load, current_loads)
            assignments[assigned_server].append(task_load)

            # Update current loads for next assignment
            current_loads[assigned_server] += task_load

        return assignments

    def get_load_distribution(self, assignments: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculate the total load assigned to each server.

        Parameters:
        - assignments: Task assignments from balance_workload

        Returns:
        - Dictionary of total loads per server
        """
        return {server: sum(tasks) for server, tasks in assignments.items()}

def simulate_balancing(predictors: Dict[str, 'LSTMWorkloadPredictor'], num_tasks: int = 20):
    """
    Simulate the workload balancing process.

    Parameters:
    - predictors: Trained LSTM predictors
    - num_tasks: Number of tasks to simulate
    """
    balancer = SmartWorkloadBalancer(predictors, len(predictors))

    # Initialize current loads
    current_loads = {server_id: np.random.uniform(20, 50) for server_id in predictors.keys()}

    # Initialize historical loads
    for server_id in predictors.keys():
        balancer.server_loads[server_id] = list(np.random.uniform(30, 60, predictors[server_id].lookback))

    # Generate incoming tasks
    incoming_tasks = np.random.uniform(5, 15, num_tasks)

    print("Initial loads:", current_loads)
    print(f"Incoming tasks: {incoming_tasks}")

    # Balance workload
    assignments = balancer.balance_workload(incoming_tasks, current_loads)
    final_loads = balancer.get_load_distribution(assignments)

    print("Task assignments:", assignments)
    print("Final load distribution:", final_loads)

    # Calculate load variance (lower is better)
    loads = list(final_loads.values())
    load_variance = np.var(loads)
    print(f"Load variance: {load_variance:.2f}")

if __name__ == "__main__":
    # Example usage (requires trained predictors)
    print("Workload Balancer module loaded. Use simulate_balancing() with trained predictors.")
