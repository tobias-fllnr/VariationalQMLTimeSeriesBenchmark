import pandas as pd
import torch
import numpy as np

class DataHandling:
    """Class to handle time series data loading, normalization, and splitting for training/testing."""

    def __init__(self, data_label, seq_length, prediction_step):
        """
        Initialize DataHandling class.

        Parameters:
        - data_label_entry: Identifier for the dataset to be used.
        - seq_length: Length of input sequences for the model.
        - prediction_step: How many time steps ahead the model should predict.
        """

        self.data_label_entry = data_label
        self.seq_length = seq_length
        self.prediction_step = prediction_step

        # Predefined dataset metadata: file paths, lengths, and split sizes
        data_info = {
            "mackey_1000": {"file_path": "./TimeseriesData/mackey_1000.csv", "data_length": 1000, "validation_size": 0.2, "test_size": 0.2},
            "henon_1000":  {"file_path": "./TimeseriesData/henon_1000.csv", "data_length": 1000, "validation_size": 0.2, "test_size": 0.2},
            "lorenz_1000": {"file_path": "./TimeseriesData/lorenz_1000.csv", "data_length": 1000, "validation_size": 0.2, "test_size": 0.2},
        }

        # Load dataset metadata if label is found
        if self.data_label_entry in data_info:
            self.file_path = data_info[self.data_label_entry]["file_path"]
            self.data_length = data_info[self.data_label_entry]["data_length"]
            self.validation_size = data_info[self.data_label_entry]["validation_size"]
            self.test_size = data_info[self.data_label_entry]["test_size"]
        else:
            raise ValueError("Data label not found in data_info")

        # Load the actual data and keep min/max values for normalization
        self.data, self.min_values, self.max_values = self.load_data()
    
    def load_data(self):
        """
        Load data from CSV file and compute min/max values per column.

        Returns:
        - data: Pandas DataFrame containing the loaded data
        - min_values: List of minimum values per column (for normalization)
        - max_values: List of maximum values per column
        """
        data = pd.read_csv(self.file_path)
        data = data.head(self.data_length)  # Truncate to specified length
        min_values = []
        max_values = []
        for column in data.columns:
            min_values.append(data[column].min())
            max_values.append(data[column].max())
        return data, min_values, max_values

    def transform(self):
        """
        Normalize each column in the data to the [0, 1] range.

        Returns:
        - Normalized data (Pandas DataFrame)
        """
        data = self.data.copy()
        for i, column in enumerate(self.data.columns):
            data[column] = (data[column] - self.min_values[i]) / (self.max_values[i] - self.min_values[i])
        return data
    
    def inverse_transform(self, data):
        """
        Revert the normalization back to original data scale.

        Parameters:
        - data: Pandas DataFrame with normalized values

        Returns:
        - DataFrame with original scale values
        """
        for i, column in enumerate(data.columns):
            data[column] = data[column] * (self.max_values[i] - self.min_values[i]) + self.min_values[i]
        return data

    def get_training_and_test_data(self):
        """
        Split the time series into training, validation, and test sets using sliding windows.

        Returns:
        - inputs_training: Tensor of input sequences for training
        - labels_training: Tensor of corresponding target values
        - inputs_validation: Tensor for validation input sequences
        - labels_validation: Corresponding targets
        - inputs_testing: Tensor for testing input sequences
        - labels_testing: Corresponding targets
        """
        data = self.transform()
        x = []  # List to hold input sequences
        y = []  # List to hold corresponding targets

        # Construct input-output pairs with sliding window
        for i in range(len(data) - self.seq_length - self.prediction_step):
            x.append(data.iloc[i:i+self.seq_length].values)  # Input sequence
            y.append(data.iloc[i+self.seq_length+self.prediction_step-1].values)  # Prediction target

        # Calculate indices to split into train/validation/test
        split_index_validation = int(len(x) * (1 - self.test_size - self.validation_size))
        split_index_test = int(len(x) * (1 - self.test_size))

        # Convert to NumPy arrays
        x, y = np.array(x), np.array(y)

        # Convert to PyTorch tensors and split according to calculated indices
        inputs_training = torch.tensor(x[:split_index_validation], dtype=torch.float32)
        inputs_validation = torch.tensor(x[split_index_validation:split_index_test], dtype=torch.float32)
        inputs_testing = torch.tensor(x[split_index_test:], dtype=torch.float32)

        labels_training = torch.tensor(y[:split_index_validation], dtype=torch.float32)
        labels_validation = torch.tensor(y[split_index_validation:split_index_test], dtype=torch.float32)
        labels_testing = torch.tensor(y[split_index_test:], dtype=torch.float32)

        return inputs_training, labels_training, inputs_validation, labels_validation, inputs_testing, labels_testing
