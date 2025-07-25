from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import os
import json 
import itertools
import models as models
from handling_data import DataHandling
from trainer import Trainer
from analyzer import Analyzer


def load_json_file(config_path: str, version: str, model_name: str) -> dict:
    """
    Loads a JSON file, creates a directory for analyzed configurations, and writes the data to a new JSON file.

    Args:
        config_path (str): Path to the JSON configuration file.
        version (str): Version identifier for the analyzed configuration.
        model_name (str): Name of the model.

    Returns:
        dict: The loaded JSON data.

    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
    """
    # Define the directory path
    path = config_path
    # Load the JSON file
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    with open(path, "r") as file:
        data = json.load(file)

    path_trained = f"./Analyzed_Configurations/Version_{version}/{model_name}"
    if not os.path.exists(path_trained):
        os.makedirs(path_trained)
    # Write the data as a JSON file in the path_trained
    parts = os.path.normpath(config_path).split(os.sep)
    submission_number = parts[-1].split(".")[0]
    output_path = os.path.join(path_trained, f"{submission_number}.json")
    with open(output_path, "w") as outfile:
        json.dump(data, outfile, indent=4)
    return data


def generate_combinations(param_dict: Dict[str, List]) -> List[Dict[str, any]]:
    """
    Generates all possible combinations of parameters from a dictionary of parameter lists.

    Args:
        param_dict (Dict[str, List]): A dictionary where keys are parameter names and values are lists of parameter values.

    Returns:
        List[Dict[str, any]]: A list of dictionaries, each representing a unique combination of parameters.
    """
    keys, values = zip(*param_dict.items())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return combinations


def extract(config: Dict[str, Any], model_name: str, version: str) -> Tuple:
    """
    Extracts parameters from a configuration dictionary and initializes a model based on the model name.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing parameter values.
        model_name (str): Name of the model to initialize (e.g., "vqc").
        version (str): Version identifier for the analyzed configuration.

    Returns:
        Tuple: A tuple containing extracted parameters and metrics.
    """
    # Extract parameters
    random_id = config["random_ids"]
    data_label = config["data_labels"]
    learning_rate = config["learning_rates"]
    num_qubits = config["num_qubits"]
    hidden_size = config["hidden_sizes"]
    ansatz = config["ansatz_types"]
    seq_length = config["sequence_lengths"]
    prediction_step = config["prediction_steps"]
    batch_size = config["batch_sizes"]
    if model_name == "vqc":
        model = models.VQC(
            num_qubits=num_qubits,
            seq_length=seq_length,
            ansatz=ansatz,
            data_label=data_label,
            random_id=random_id,
            evaluation=True,
        )
    elif model_name == "qlstm_paper":
        model = models.QLSTM_Paper(
            num_qubits=num_qubits, ansatz=ansatz, data_label=data_label, random_id=random_id
        )
    elif model_name == "qlstm_linear_enhanced_paper":
        model = models.QLSTM_Linerar_Enhanced_Paper(
            num_qubits=num_qubits,
            hidden_size=hidden_size,
            ansatz=ansatz,
            data_label=data_label,
            random_id=random_id,
        )
    elif model_name == "qrnn_paper":
        model = models.QRNN_Paper(
            num_qubits=num_qubits,
            num_qubits_hidden=hidden_size,
            seq_length=seq_length,
            ansatz=ansatz,
            data_label=data_label,
            random_id=random_id,
        )
    elif model_name == "lstm":
        model = models.LSTM(
            hidden_size=hidden_size, ansatz=ansatz, data_label=data_label, random_id=random_id
        )
    elif model_name == "rnn":
        model = models.RNN(
            hidden_size=hidden_size, ansatz=ansatz, data_label=data_label, random_id=random_id
        )
    elif model_name == "mlp":
        model = models.MLP(
            seq_length=seq_length, ansatz=ansatz, data_label=data_label, random_id=random_id
        )
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
    data_handler = DataHandling(
        data_label=data_label, seq_length=seq_length, prediction_step=prediction_step
    )
    trainer = Trainer(
        model=model, random_id=random_id, learning_rate=learning_rate, batch_size=batch_size
    )
    analyzer = Analyzer(version=version, model=model, trainer=trainer, data_handler=data_handler)
    if analyzer.load_model():
        loss_metrics = pd.read_csv(analyzer.path + "/loss_metrics.csv")
        training_info = pd.read_csv(analyzer.path + "/training_info.csv")
        mse_testing = loss_metrics["MSE Testing"][0]
        mse_validation = loss_metrics["MSE Validation"][0]
        mae_testing = loss_metrics["MAE Testing"][0]
        mae_validation = loss_metrics["MAE Validation"][0]
        corr_testing = loss_metrics["Correlation Testing"][0]
        corr_validation = loss_metrics["Correlation Validation"][0]
        num_parameters = analyzer.get_number_of_parameters()
        epochs_to_convergance = training_info["Epochs to Convergence"][0]
        total_training_time = training_info["Total Training Time"][0]
        training_loss_100_epochs = training_info["Training Loss after 100 epochs"][0]
        validation_loss_100_epochs = training_info["Validation Loss after 100 epochs"][0]
        testing_loss_100_epochs = training_info["Testing Loss after 100 epochs"][0]
        print(
            f"Model: {model_name}, Data: {data_label}, Random ID: {random_id}, Learning Rate: {learning_rate}, Num Qubits: {num_qubits}, Hidden Size: {hidden_size}, Ansatz: {ansatz}, Sequence Length: {seq_length}, Prediction Step: {prediction_step}, Batch Size: {batch_size}"
        )
        tupel = (
            version,
            model_name,
            ansatz,
            data_label,
            random_id,
            learning_rate,
            num_qubits,
            hidden_size,
            seq_length,
            prediction_step,
            batch_size,
            mse_testing,
            mse_validation,
            mae_testing,
            mae_validation,
            corr_testing,
            corr_validation,
            num_parameters,
            epochs_to_convergance,
            total_training_time,
            training_loss_100_epochs,
            validation_loss_100_epochs,
            testing_loss_100_epochs,
        )
        return tupel


def average_random_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Averages metrics across random IDs in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing metrics and random IDs.

    Returns:
        pd.DataFrame: DataFrame with averaged metrics.
    """
    groupby_columns = [
        "Version",
        "Model",
        "Ansatz",
        "Data",
        "Learning Rate",
        "Number Qubits",
        "Hidden Size",
        "Sequence Length",
        "Prediction Step",
        "Batch Size",
    ]
    metrics = [
        "MSE Testing",
        "MSE Validation",
        "MAE Testing",
        "MAE Validation",
        "Correlation Testing",
        "Correlation Validation",
        "Epochs to Convergance",
        "Total Training Time",
        "Training Loss after 100 epochs",
        "Validation Loss after 100 epochs",
        "Testing Loss after 100 epochs",
    ]
    aggregations = ["mean", "std", "min", "max", "median", "mad"]

    # Define a function to compute MAD (median absolute deviation)
    def mad(series: pd.Series) -> float:
        median = np.median(series)
        return np.median(np.abs(series - median))

    # Extend pandas with MAD if not available
    if "mad" not in pd.Series.__dict__:
        pd.Series.mad = mad

    # Create a dictionary for statistical functions
    agg_funcs = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
        "median": np.median,
        "mad": mad,
    }

    # Group the DataFrame
    grouped = df.groupby(groupby_columns)
    # Compute statistics for each group
    results = []
    for _, indices in grouped.groups.items():
        group_df = df.loc[indices]
        stats = []
        for metric in metrics:
            for agg in aggregations:
                stats.append(agg_funcs[agg](group_df[metric]))

        # Include group identifiers and "Num Parameters" from the first row
        first_row = group_df.iloc[0][groupby_columns + ["Num Parameters"]]
        results.append([*first_row, *stats])

    # Define column names dynamically
    metric_columns = [f"{metric} {agg.capitalize()}" for metric in metrics for agg in aggregations]
    result_columns = groupby_columns + ["Num Parameters"] + metric_columns

    # Create the resulting DataFrame
    stats_df = pd.DataFrame(results, columns=result_columns)
    return stats_df


def hyperparameter_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs hyperparameter optimization by selecting the best configurations.

    Args:
        df (pd.DataFrame): Input DataFrame containing metrics and configurations.

    Returns:
        pd.DataFrame: DataFrame with the best configurations for each group.
    """
    hyper_opt_df = df[
        df["MSE Validation Mad"] > 0
    ]  # to ensure that the MAD is not zero which is the case when there is only training combination for this hyperparameter combination
    groupby_columns_lr = [
        "Version",
        "Model",
        "Data",
        "Prediction Step",
        "Ansatz",
        "Sequence Length",
        "Number Qubits",
        "Hidden Size",
    ]

    # Group the dataframe
    grouped_lr = hyper_opt_df.groupby(groupby_columns_lr, dropna=False)

    # Find the index of the row with the minimum 'MSE Validation Medium' in each group
    min_mse_idx = grouped_lr["MSE Validation Median"].idxmin()

    # Create a new dataframe with the rows having the minimum 'MSE Validation Medium' for each group
    min_mse_df = hyper_opt_df.loc[min_mse_idx]

    # Reset index if needed
    min_mse_df.reset_index(drop=True, inplace=True)
    return min_mse_df


if __name__ == "__main__":

    version = 1
    submittet_configs_paths = []
    for dirpath, _, filenames in os.walk(f"./Submitted_Configurations/Version_{version}"):
        for filename in filenames:
            if filename.endswith('.json'):
                full_path = os.path.join(dirpath, filename)
                submittet_configs_paths.append(full_path)

    tupel_list = []
    for config_path in submittet_configs_paths:
        parts = os.path.normpath(config_path).split(os.sep)
        model_name = parts[-2]
        file_path = f"./Results/{model_name}_results.csv"
        configurations = load_json_file(config_path, version, model_name)
        combinations = generate_combinations(configurations)
        for combo in combinations:
            tupel = extract(combo, model_name, version)
            tupel_list.append(tupel)

        df_new = pd.DataFrame(
            tupel_list,
            columns=[
                "Version",
                "Model",
                "Ansatz",
                "Data",
                "Random ID",
                "Learning Rate",
                "Number Qubits",
                "Hidden Size",
                "Sequence Length",
                "Prediction Step",
                "Batch Size",
                "MSE Testing",
                "MSE Validation",
                "MAE Testing",
                "MAE Validation",
                "Correlation Testing",
                "Correlation Validation",
                "Num Parameters",
                "Epochs to Convergance",
                "Total Training Time",
                "Training Loss after 100 epochs",
                "Validation Loss after 100 epochs",
                "Testing Loss after 100 epochs",
            ],
        )
        if os.path.exists(file_path):
            df_old = pd.read_csv(file_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(file_path, index=False)
        df = pd.read_csv(file_path)
        averaged_ids_df = average_random_ids(df)
        averaged_ids_df.to_csv(f"./Results/{model_name}_averaged_ids.csv", index=False)
        hyper_opt_df = hyperparameter_optimization(averaged_ids_df)
        hyper_opt_df.to_csv(f"./Results/{model_name}_hyper_opt.csv", index=False)
