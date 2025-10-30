import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

class LSTMWorkloadPredictor:
    def __init__(self, lookback=10, epochs=50, batch_size=32):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def build_model(self, input_shape):
        """
        Build the LSTM model architecture.

        Parameters:
        - input_shape: Shape of input data (timesteps, features)
        """
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X_train, y_train, validation_split=0.2):
        """
        Train the LSTM model.

        Parameters:
        - X_train: Training input sequences
        - y_train: Training target values
        - validation_split: Fraction of data to use for validation
        """
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))

        # Scale the target values
        y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1))

        history = self.model.fit(
            X_train, y_train_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=1
        )

        return history

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X: Input sequences for prediction

        Returns:
        - Predicted values (inverse scaled)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        predictions_scaled = self.model.predict(X)
        predictions = self.scaler.inverse_transform(predictions_scaled)

        return predictions.flatten()

    def save_model(self, filepath):
        """
        Save the trained model and scaler.

        Parameters:
        - filepath: Base filepath for saving (without extension)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        self.model.save(f"{filepath}.h5")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")

    def load_model(self, filepath):
        """
        Load a trained model and scaler.

        Parameters:
        - filepath: Base filepath for loading (without extension)
        """
        self.model = tf.keras.models.load_model(f"{filepath}.h5")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")

def create_server_predictors(df, lookback=10):
    """
    Create separate LSTM predictors for each server.

    Parameters:
    - df: DataFrame with workload data
    - lookback: Number of previous time steps for prediction

    Returns:
    - Dictionary of predictors keyed by server_id
    """
    from data_generator import prepare_lstm_data

    predictors = {}
    for server_id in df['server_id'].unique():
        server_df = df[df['server_id'] == server_id]
        X, y = prepare_lstm_data(server_df, lookback)

        predictor = LSTMWorkloadPredictor(lookback=lookback)
        predictor.train(X, y)

        predictors[server_id] = predictor

    return predictors

if __name__ == "__main__":
    # Example usage
    from data_generator import generate_synthetic_workload_data, prepare_lstm_data

    df = generate_synthetic_workload_data(num_servers=1, num_timesteps=500)
    X, y = prepare_lstm_data(df)

    predictor = LSTMWorkloadPredictor()
    history = predictor.train(X, y)

    predictions = predictor.predict(X[:10])  # Predict first 10 sequences
    print(f"Sample predictions: {predictions}")
