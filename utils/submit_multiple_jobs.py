import itertools
import os
import json
from typing import Dict, List, Any
import argparse


def generate_combinations(param_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all possible combinations of parameters from a dictionary of parameter lists.

    Args:
        param_dict (Dict[str, List[Any]]): Dictionary where keys are parameter names and values are
            lists of possible values.

    Returns:
        List[Dict[str, Any]]: List of dictionaries, each representing a unique combination
            of parameters.
    """
    keys, values = zip(*param_dict.items())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return combinations


def submit_job(job_name: str, memory: int, command: str, script_filename: str, combo: Any) -> None:
    """
    Submits a job to a SLURM cluster by generating a submission script and executing it.

    Args:
        job_name (str): The name of the job to be submitted.
        memory (int): The amount of memory (in GB) to allocate for the job.
        command (str): The shell command to execute within the SLURM job.
        script_filename (str): The filename for the temporary SLURM submission script.
        combo (Any): The parameter combination or identifier for the job, used for logging.

    Returns:
        None
    """
    # SLURM submission script template
    slurm_template = """#!/bin/bash
    #SBATCH --job-name={job_name}
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task=1
    #SBATCH --mem={memory}GB
    #SBATCH --time=48:00:00

    {command}
    """
    slurm_script = slurm_template.format(job_name=job_name, memory=memory, command=command)

    with open(script_filename, "w") as script_file:
        script_file.write(slurm_script)

    # Submit the job to the SLURM cluster
    os.system(f"sbatch {script_filename}")
    print(f"Job submitted for combination: {combo}")

    # delete the sh file
    os.remove(script_filename)


def determine_memory(model_name: str, num_qubits: int, ansatz_type: str) -> int:
    """
    Determines the amount of memory required for a job based on the model, number of qubits, and
        ansatz type.

    Args:
        model_name (str): Name of the model.
        num_qubits (int): Number of qubits.
        ansatz_type (str): Type of ansatz.

    Returns:
        int: Amount of memory in GB.
    """
    memory = 0
    if model_name in ["vqc", "qlstm_paper"]:
        if num_qubits <= 6:
            memory = 2
        elif num_qubits <= 8:
            memory = 4
        elif num_qubits <= 10:
            memory = 8
        elif num_qubits <= 16:
            memory = 32
    elif model_name in ["qlstm_linear_enhanced_paper"]:
        memory = 4
    elif model_name in ["qrnn_paper"]:
        if ansatz_type == "paper_reset":
            if num_qubits <= 4:
                memory = 8
            else:
                memory = 64
        elif ansatz_type == "paper_no_reset":
            memory = 4
    elif model_name in ["lstm", "rnn", "mlp"]:
        memory = 2
    return memory


def run_experiment(config: Dict[str, Any], slurm) -> Any:
    """
    Runs an experiment by extracting parameters from the configuration, preparing the SLURM job,
    and submitting it.

    Args:
        config (Dict[str, Any]): Dictionary containing experiment configuration.

    Returns:
        Any: Tuple containing version and model_name.
    """
    # Extract parameters
    version = config["version"]
    model_name = config["model_names"]
    random_id = config["random_ids"]
    data_label = config["data_labels"]
    learning_rate = config["learning_rates"]
    num_qubits = config["num_qubits"]
    hidden_size = config["hidden_sizes"]
    ansatz_type = config["ansatz_types"]
    sequence_length = config["sequence_lengths"]
    prediction_step = config["prediction_steps"]
    batch_size = config["batch_sizes"]

    if slurm:

        job_name = f"{version}_{model_name}_{data_label}_{random_id}_{learning_rate}_{num_qubits}_{hidden_size}_{ansatz_type}_{sequence_length}_{prediction_step}_{batch_size}"
        command = f"srun python3 training_and_analyzing.py -version {version} -model {model_name} -data {data_label} -id {random_id} -lr {learning_rate} -ansatz {ansatz_type} -seq_length {sequence_length} -pred_step {prediction_step} -num_qubits {num_qubits} -hidden_size {hidden_size} -batch_size {batch_size}"
        script_filename = f"submit_{version}_{model_name}_{data_label}_{random_id}_{learning_rate}_{num_qubits}_{ansatz_type}_{prediction_step}_{sequence_length}_{batch_size}.sh"
        memory = determine_memory(model_name, num_qubits, ansatz_type)

        submit_job(job_name, memory, command, script_filename, config)
    
    else:
        # Run the command directly without SLURM
        command = f"python3 training_and_analyzing.py -version {version} -model {model_name} -data {data_label} -id {random_id} -lr {learning_rate} -ansatz {ansatz_type} -seq_length {sequence_length} -pred_step {prediction_step} -num_qubits {num_qubits} -hidden_size {hidden_size} -batch_size {batch_size}"
        os.system(command)

    return version, model_name


def save_training_configuration(item: Dict[str, Any]) -> None:
    """
    Saves the training configuration as a JSON file in a structured directory.

    Args:
        item (Dict[str, Any]): The configuration dictionary to save.

    Returns:
        None
    """
    version = item["version"][0]
    model_name = item["model_names"][0]
    # Define the directory path
    path = f"../Submitted_Configurations/Version_{version}/{model_name}"

    # Ensure the directory exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Find the next available file name
    i = 1
    while True:
        filename = os.path.join(path, f"{i}.json")  # Full path to the file
        if not os.path.exists(filename):  # Check if the file exists
            # Save the JSON file
            with open(filename, "w") as output_file:
                json.dump(item, output_file, indent=4)
            break
        i += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Submission Controller")
    parser.add_argument(
        "-model", "--model", type=str, help="models to train")
    parser.add_argument(
        "-slurm", "--slurm", type="str", help="yes or no")
    
    args = parser.parse_args()
    model = args.model
    slurm = args.slurm
    if slurm == "yes":
        slurm = True
    else:
        slurm = False

    path_configurations = f"../Training_Configurations/configurations_{model}.json"

    try:
        # Read the JSON file
        with open(path_configurations, "r") as file:
            data = json.load(file)

        if not isinstance(data, list):
            print("The JSON file does not contain a list of dictionaries.")

        # Process each element in the list
        for item in data[:]:  # Iterate over a copy to allow modification
            try:
                # Load the item as a dictionary
                if isinstance(item, dict):
                    print("Processing:", item)
                    # Perform your processing logic here

                    combinations = generate_combinations(item)
                    for combo in combinations:
                        run_experiment(combo, slurm)

                    # Save the training configuration to ../Submitted_Configurations
                    save_training_configuration(item)

                else:
                    raise ValueError("Item is not a dictionary")
            except Exception as e:
                print(f"Error processing item: {item}. Error: {e}")

    except FileNotFoundError:
        print(f"The file {path_configurations} does not exist.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in the file {path_configurations}.")

    print("All jobs submitted.")
