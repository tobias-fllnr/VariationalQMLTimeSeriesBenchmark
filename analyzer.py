import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-deep")


class Analyzer:
    """
    class to analyse the model after training
    """
    def __init__(self, version, model, trainer, data_handler):
        self.model = model
        self.trainer = trainer
        self.data_handler = data_handler
        if self.model._get_name() in ["VQC"]:
            self.path = f"./Data/Version_{version}/{self.model._get_name()}/{self.data_handler.data_label}/{self.model.ansatz}/Sequence_length_{self.data_handler.seq_length}/Prediction_step_{self.data_handler.prediction_step}/Num_qubits_{self.model.num_qubits}/ID_{self.model.random_id}_lr_{self.trainer.learning_rate}_batch_{self.trainer.batch_size}"
        elif self.model._get_name() in ["QLSTM_Paper"]:
            self.path = f"./Data/Version_{version}/{self.model._get_name()}/{self.data_handler.data_label}/{self.model.ansatz}/Sequence_length_{self.data_handler.seq_length}/Prediction_step_{self.data_handler.prediction_step}/Num_qubits_{self.model.num_qubits}/ID_{self.model.random_id}_lr_{self.trainer.learning_rate}_batch_{self.trainer.batch_size}"
        elif self.model._get_name() in ["QLSTM_Linerar_Enhanced_Paper"]:
            self.path = f"./Data/Version_{version}/{self.model._get_name()}/{self.data_handler.data_label}/{self.model.ansatz}/Sequence_length_{self.data_handler.seq_length}/Prediction_step_{self.data_handler.prediction_step}/Num_qubits_{self.model.num_qubits}/Hidden_size_{self.model.hidden_size}/ID_{self.model.random_id}_lr_{self.trainer.learning_rate}_batch_{self.trainer.batch_size}"
        elif self.model._get_name() in ["QRNN_Paper"]:
            self.path = f"./Data/Version_{version}/{self.model._get_name()}/{self.data_handler.data_label}/{self.model.ansatz}/Sequence_length_{self.data_handler.seq_length}/Prediction_step_{self.data_handler.prediction_step}/Num_qubits_{self.model.num_qubits}/Num_qubits_data_{self.model.num_qubits_data}/ID_{self.model.random_id}_lr_{self.trainer.learning_rate}_batch_{self.trainer.batch_size}"
        elif self.model._get_name() in ["LSTM", "RNN"]:
            self.path = f"./Data/Version_{version}/{self.model._get_name()}/{self.data_handler.data_label}/{self.model.ansatz}/Sequence_length_{self.data_handler.seq_length}/Prediction_step_{self.data_handler.prediction_step}/Hidden_size_{self.model.hidden_size}/ID_{self.model.random_id}_lr_{self.trainer.learning_rate}_batch_{self.trainer.batch_size}"
        elif self.model._get_name() in ["MLP"]:
            self.path = f"./Data/Version_{version}/{self.model._get_name()}/{self.data_handler.data_label}/{self.model.ansatz}/Sequence_length_{self.data_handler.seq_length}/Prediction_step_{self.data_handler.prediction_step}/ID_{self.model.random_id}_lr_{self.trainer.learning_rate}_batch_{self.trainer.batch_size}"
    def create_directory(self):
        """
        create the directory to store all data and plots of the specific training
        """
        # Check if the directory already exists
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        else:
            pass

    def load_model(self):
        """
        checks if directory exists. If exists, return true and load model, if not return false
        :return:
        """
        if os.path.exists(self.path + "/trained_model"):
            self.model.load_state_dict(torch.load(self.path + "/trained_model", weights_only=True))
            self.model.eval()
            return True
        else:
            return False
    
    def save_training_output(self,trained_model, best_validation_model, cost_training, cost_validation, cost_testing):
        """
        save outputs of training
        """
        torch.save(trained_model, self.path + "/trained_model")
        torch.save(best_validation_model, self.path + "/best_validation_model")
        np.save(self.path + "/cost_training", cost_training)

    def plot_cost(self, cost_training, cost_validation, cost_testing):
        """
        plot cost over epochs
        """
        plt.figure()
        plt.plot(cost_training, label="Training")
        plt.plot(cost_validation, label="Validation")
        plt.plot(cost_testing, label="Testing")
        plt.xlabel("Epoch")
        plt.ylabel("Cost (MSE)")
        plt.yscale("log")
        plt.legend()
        plt.savefig(self.path + "/cost_plot.pdf")
        plt.close()
    
    def evaluate_trained_model(self, inputs_testing, labels_testing, inputs_validation, labels_validation):
        """
        evaluate the trained model
        """

        def calculate_values(output_validation, output_testing):
            mse_testing = nn.MSELoss()(output_testing, labels_testing).item()
            mse_validation = nn.MSELoss()(output_validation, labels_validation).item()
            mae_testing = nn.L1Loss()(output_testing, labels_testing).item()
            mae_validation = nn.L1Loss()(output_validation, labels_validation).item()

            corr_test = torch.stack((output_testing, labels_testing))
            corr_test = torch.reshape(corr_test, (corr_test.size(0), -1))
            corr_testing = torch.corrcoef(corr_test)[0][1].item()
            corr_validation = torch.stack((output_validation, labels_validation))
            corr_validation = torch.reshape(corr_validation, (corr_validation.size(0), -1))
            corr_validation = torch.corrcoef(corr_validation)[0][1].item()
            data = {
                "MSE Testing": mse_testing,
                "MSE Validation": mse_validation,
                "MAE Testing": mae_testing,
                "MAE Validation": mae_validation,
                "Correlation Testing": corr_testing,
                "Correlation Validation": corr_validation
            }
            return pd.DataFrame(data, index=[0])
        self.model.load_state_dict(torch.load(self.path + "/trained_model", weights_only=True))
        self.model.eval()
        output_testing = self.model(inputs_testing)
        output_validation = self.model(inputs_validation)
        df_last = calculate_values(output_validation, output_testing)
        df_last.to_csv(self.path + "/loss_metrics_end.csv", index=False)
        self.model.load_state_dict(torch.load(self.path + "/best_validation_model", weights_only=True))
        self.model.eval()
        output_testing = self.model(inputs_testing)
        output_validation = self.model(inputs_validation)
        df_best = calculate_values(output_validation, output_testing)
        df_best.to_csv(self.path + "/loss_metrics.csv", index=False)


    def save_training_info(self, cost_training, cost_validation, cost_testing, total_time):
        """
        Save the number of epochs to convergence and the total training time in a CSV file.
        """
        num_epochs = len(cost_training)
        training_loss_100_epochs = cost_training[99]
        validation_loss_100_epochs = cost_validation[99]
        testing_loss_100_epochs = cost_testing[99]

        data = {
            "Metric": ["Epochs to Convergence", "Total Training Time", "Training Loss after 100 epochs", "Validation Loss after 100 epochs", "Testing Loss after 100 epochs"],
            "Value": [num_epochs, total_time, training_loss_100_epochs, validation_loss_100_epochs, testing_loss_100_epochs]
        }
        data = {
                "Epochs to Convergence": num_epochs,
                "Total Training Time": total_time,
                "Training Loss after 100 epochs": training_loss_100_epochs,
                "Validation Loss after 100 epochs": validation_loss_100_epochs,
                "Testing Loss after 100 epochs": testing_loss_100_epochs
            }

        df = pd.DataFrame(data, index=[0])
        df.to_csv(self.path + "/training_info.csv", index=False)
        
    def get_number_of_parameters(self):
        """
        get the number of parameters of the model
        """
        total_parameters = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total_parameters += param.numel()
        return total_parameters