import copy
import time
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import shuffle


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        random_id: int = 42,
        learning_rate: float = 0.1,
        print_gradients: bool = False,
        batch_size: int = 64
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The model to be trained.
            random_id (int, optional): Random seed for reproducibility. Defaults to 42.
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.1.
            print_gradients (bool, optional): Whether to print gradients during training. Defaults to False.
            batch_size (int, optional): Batch size for training. Defaults to 64.
        """
        super(Trainer, self).__init__()
        self.model = model
        self.random_id = random_id
        self.learning_rate = learning_rate
        self.print_gradients = print_gradients
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.cost = nn.MSELoss()

    def train(
        self,
        inputs_training: torch.Tensor,
        labels_training: torch.Tensor,
        inputs_validation: torch.Tensor,
        labels_validation: torch.Tensor,
        inputs_testing: torch.Tensor,
        labels_testing: torch.Tensor,
    ) -> Tuple[
        List[float], List[float], List[float], Any, Any, float
    ]:
        """
        Train the model using the provided data.

        Args:
            inputs_training (torch.Tensor): Training input data.
            labels_training (torch.Tensor): Training labels.
            inputs_validation (torch.Tensor): Validation input data.
            labels_validation (torch.Tensor): Validation labels.
            inputs_testing (torch.Tensor): Testing input data.
            labels_testing (torch.Tensor): Testing labels.

        Returns:
            Tuple containing:
                - cost_training (List[float]): Training loss per epoch.
                - cost_validation (List[float]): Validation loss per epoch.
                - cost_testing (List[float]): Testing loss per epoch.
                - model_end (Any): Model state dict at end of training.
                - model_best_validation (Any): Model state dict with best validation loss.
                - total_time (float): Total training time in seconds.
        """
        time_start = time.time()
        cost_training = []
        cost_validation = []
        cost_testing = []
        converged = False
        min_validation_loss = np.inf
        i = 0
        while not converged:
            i += 1
            time_epoch_start = time.time()
            inputs_training, labels_training = shuffle(
                inputs_training, labels_training, random_state=self.random_id
            )
            total_loss_training = 0
            num_batches = 0
            for j in range(0, len(inputs_training), self.batch_size):
                inputs_batch = inputs_training[j : j + self.batch_size]
                labels_batch = labels_training[j : j + self.batch_size]
                self.optimizer.zero_grad()
                output_training = self.model(inputs_batch)
                loss_training = self.cost(output_training, labels_batch)
                loss_training.backward()
                self.optimizer.step()

                if self.print_gradients:
                    print(f"Gradients for epoch {i}:")
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            print(f"\t{name}:")
                            print(param.grad)
                        else:
                            print(f"\t{name}: No gradient")
                total_loss_training += loss_training.item()
                # total_loss_validation += loss_validation.item()
                # total_loss_testing += loss_testing.item()
                num_batches += 1
            avg_loss_training = total_loss_training / num_batches
            cost_training.append(avg_loss_training)
            with torch.no_grad():
                output_validation = self.model(inputs_validation)
                loss_validation = self.cost(output_validation, labels_validation).item()

                cost_validation.append(loss_validation)
                output_testing = self.model(inputs_testing)
                loss_testing = self.cost(output_testing, labels_testing)
                cost_testing.append(loss_testing.item())
            if loss_validation < min_validation_loss:
                model_best_validation = copy.deepcopy(self.model.state_dict())
                min_validation_loss = loss_validation

            time_epoch_end = time.time()
            print(
                f"Epoch {i}: loss training={round(cost_training[-1], 7)},"
                f"loss validation={round(cost_validation[-1], 7)},"
                f"loss testing={round(cost_testing[-1], 7)},"
                f"time for epoch={round(time_epoch_end-time_epoch_start, 2)}",
                flush=True,
            )
            if self.check_convergence(cost_validation):
                converged = True
                print(f"model converged", flush=True)

        time_end = time.time()
        total_time = round(time_end - time_start, 2)
        print(f"Training finished! Total time={total_time}", flush=True)
        model_end = self.model.state_dict()
        return (
            cost_training,
            cost_validation,
            cost_testing,
            model_end,
            model_best_validation,
            total_time,
        )

    def check_convergence(self, cost_validation: List[float]) -> bool:
        """
        Check if the model has converged based on validation cost.

        Args:
            cost_validation (List[float]): List of validation costs.

        Returns:
            bool: True if the model has converged, False otherwise.
        """
        if len(cost_validation) < 400:
            return False
        else:
            mean1 = np.mean(cost_validation[-400:-200])
            mean2 = np.mean(cost_validation[-200:])
            derivation2 = np.std(cost_validation[-200:])
            return np.abs(mean1 - mean2) < derivation2 / 2
