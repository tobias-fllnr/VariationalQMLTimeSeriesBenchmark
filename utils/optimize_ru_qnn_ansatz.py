import pandas as pd
import itertools
import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple, Optional
import models
from handling_data import DataHandling
from trainer import Trainer
from analyzer import Analyzer


model_name = "vqc"
version = 1

def load_json_file(model_name: str, version: int, submission_number: int) -> Dict[str, Any]:
    """
    Loads a JSON file, creates a directory for analyzed configurations, and writes the data to a new JSON file.

    Args:
        model_name (str): Name of the model.
        version (int): Version identifier for the analyzed configuration.
        submission_number (int): Submission number of the configuration.

    Returns:
        Dict[str, Any]: The loaded JSON data.

    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
    """
    # Define the directory path
    path = f"../Submitted_Configurations/Version_{version}/{model_name}/{submission_number}.json"
    # Load the JSON file
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    with open(path, "r") as file:
        data = json.load(file)

    path_trained = f"../Analyzed_Configurations/Version_{version}/{model_name}"
    if not os.path.exists(path_trained):
        os.makedirs(path_trained)
    # Write the data as a JSON file in the path_trained
    output_path = os.path.join(path_trained, f"{submission_number}.json")
    with open(output_path, "w") as outfile:
        json.dump(data, outfile, indent=4)
    return data


def generate_combinations(param_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generates all possible combinations of parameters from a dictionary of parameter lists.

    Args:
        param_dict (Dict[str, List[Any]]): A dictionary where keys are parameter names and values are lists of parameter values.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a unique combination of parameters.
    """
    keys, values = zip(*param_dict.items())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return combinations


def extract(config: Dict[str, Any]) -> Optional[Tuple]:
    """
    Extracts parameters from a configuration dictionary, initializes a model, and retrieves metrics.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing parameter values.

    Returns:
        Optional[Tuple]: A tuple containing extracted parameters and metrics, or None if metrics are not available.
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
    model = models.VQC(
        num_qubits=num_qubits,
        seq_length=seq_length,
        ansatz=ansatz,
        data_label=data_label,
        random_id=random_id,
        evaluation=True,
    )
    data_handler = DataHandling(
        data_label=data_label, seq_length=seq_length, prediction_step=prediction_step
    )
    trainer = Trainer(
        model=model, random_id=random_id, learning_rate=learning_rate, batch_size=batch_size
    )
    analyzer = Analyzer(version=version, model=model, trainer=trainer, data_handler=data_handler)
    if analyzer.load_model():
        if not os.path.exists(analyzer.path + "/loss_metrics.csv"):
            return None
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
    

submission_numbers = [4, 5, 6]  # enter the submission numbers of the ru-QNN here
file_path = "../Results/ru_vqc_ansatz_optimization_results.csv"
tupel_list = []
for num in submission_numbers:
    configurations = load_json_file(model_name, version, num)
    combinations = generate_combinations(configurations)
    for combo in combinations:
        tupel = extract(combo)
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


df = pd.read_csv("../Results/ru_vqc_ansatz_optimization_results.csv")
groupby_columns = [
    "Version",
    "Model",
    "Data",
    "Learning Rate",
    "Hidden Size",
    "Sequence Length",
    "Prediction Step",
    "Batch Size",
]
grouped = df.groupby(groupby_columns)
output_list = []  # List to store dictionaries for JSON

with open("../Training_Configurations/configurations_best_ru_QNN.json", "w") as json_file:
    json_file.write("[\n")  # Start the JSON array
    first = True  # To manage commas between objects
    for param, indices in grouped.groups.items():
        group_df = df.loc[indices]
        group_df = group_df.nsmallest(10, "MSE Validation")

        for num_qubits in [4, 6, 8]:
            df_filtered = group_df[group_df["Number Qubits"] == num_qubits]
            ansatz_list = df_filtered["Ansatz"].tolist()
            if len(ansatz_list) > 0:
                # Create dictionary for the current group
                data_dict = {
                    "version": [1],
                    "model_names": ["vqc"],
                    "random_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    "data_labels": [param[2]],  # 'Data' value from param tuple
                    "learning_rates": [0.001],
                    "num_qubits": [num_qubits],  # 'Number Qubits' value from param tuple
                    "hidden_sizes": [None],
                    "ansatz_types": ansatz_list,
                    "sequence_lengths": [int(param[5])],  # 'Sequence Length' value from param tuple
                    "prediction_steps": [int(param[6])],  # 'Prediction Step' value from param tuple
                    "batch_sizes": [int(param[7])],  # 'Batch Size' value from param tuple
                }
                if not first:
                    json_file.write(",\n")  # Separate dictionaries with commas
                json.dump(data_dict, json_file)  # Write dictionary as JSON object
                first = False  # After the first entry, we allow commas

    json_file.write("\n]")  # Close the JSON array