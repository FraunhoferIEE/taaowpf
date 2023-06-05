import random
import sys
from typing import Tuple

from captum.robust import PGD
import pytorch_lightning as pl
import torch
from torch import nn
import torchvision

sys.path.append("/home/rheinrich/taaowpf")
from robustness_evaluation.robustness_scores import MSELossBatch


class WPF_ResNet(pl.LightningModule):
    def __init__(
        self,
        resnet_version: int,
        forecast_version: str,
        forecast_horizon: int,
        n_past_timesteps: int,
        learning_rate: float = 1e-3,
        p_adv_training: float = 0.0,
        eps_adv_training: float = 0.15,
        step_num_adv_training: int = 100,
        norm_adv_training: str = "Linf",
    ):
        """
        Initialize the WPF_ResNet model, which is used to forecast wind power across Germany.

        Args:
            resnet_version (int): Version of ResNet backbone.
            forecast_version (str): 'single' if the prediction is done step by step for single time steps, 'all' if all time steps are predicted at once.
            forecast_horizon (int): Number of future time steps to forecast.
            n_past_timesteps (int): Number of past timesteps considered for prediction.
            learning_rate (float): Learning rate for optimizer. Defaults to 1e-3.
            p_adv_training (float): Probability that input will be perturbed by adversarial attacks during training. Defaults to 0.
            eps_adv_training (float): Maximum perturbation caused by adversarial attacks. Defaults to 0.15.
            step_num_adv_training (int): Number of PGD-iterations for adversarial attacks. Defaults to 100.
            norm_adv_training (str): Norm used to calculate adversarial attacks. Defaults to 'Linf'.
        """

        super(WPF_ResNet, self).__init__()

        # Forecast horizon
        self.forecast_horizon = forecast_horizon

        # Number of past timesteps considered for prediction
        self.n_past_timesteps = n_past_timesteps

        self.forecast_version = forecast_version

        # Available ResNet versions
        resnets = {
            18: torchvision.models.resnet18,
            34: torchvision.models.resnet34,
            50: torchvision.models.resnet50,
            101: torchvision.models.resnet101,
            152: torchvision.models.resnet152,
        }

        # Using an untrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=False)

        # Replace old FC layer with identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features

        if forecast_version == "single":
            # Replace final layer for fine tuning (wind power prediction for 1 future time step)
            self.resnet_model.fc = nn.Sequential(
                nn.Linear(linear_size, 1), nn.LeakyReLU()
            )
            # We have (n_past_timesteps + 1) input channels for every wind power prediction for a single time step
            num_input_channels = n_past_timesteps + 1
        elif forecast_version == "all":
            # Replace final layer for fine tuning (wind power prediction for all future time steps)
            self.resnet_model.fc = nn.Sequential(
                nn.Linear(linear_size, forecast_horizon), nn.LeakyReLU()
            )
            # We have (n_past_timesteps + forecast_horizon) input channels if we predict all time steps at once
            num_input_channels = n_past_timesteps + forecast_horizon

        # Manually set number of channels in first Conv layer
        self.resnet_model.conv1 = nn.Conv2d(
            num_input_channels, 64, 7, stride=2, padding=3, bias=False
        )

        # Learning rate
        self.learning_rate = learning_rate

        # Probability of input being perturbed by adversarial attacks
        self.p_adv_training = p_adv_training

        # Maximum perturbation caused by adversarial attacks
        self.eps_adv_training = eps_adv_training

        # Number of PGD-iterations for adversarial attacks
        self.step_num_adv_training = step_num_adv_training

        # Norm used to calculate adversarial attacks
        self.norm_adv_training = norm_adv_training

        # Instantiate criterion for adversarial training
        self.criterion_adv_training = MSELossBatch()

        # Instantiate loss criterion
        self.criterion = nn.MSELoss()

    def forward(self, batch: torch.Tensor):
        """Performs forward pass and predicts wind power generation.

        Args:
            batch (torch.Tensor): Input batch of shape (batch_size, n_timesteps, channels, height, width).

        Returns:
            Tensor: Predictions tensor of shape (batch_size, forecast_horizon).

        Raises:
            ValueError: If forecast_version is neither 'single' nor 'all'.
        """
        # Predictions tensor of shape (forecast horizon, batch_size) filled with zeros
        if self.forecast_version == "single":
            predictions = torch.zeros(self.forecast_horizon, batch.shape[0])

            # Step-by-step prediction of wind power generation for different time steps
            for timestep in range(self.forecast_horizon):
                # Select the channels (timesteps) that are used to predict the wind power for this specific time step
                start_timestep = timestep
                end_timestep = timestep + self.n_past_timesteps + 1
                batch_timestep = batch[:, start_timestep:end_timestep, :, :]
                # Prediction for the timestep
                predictions_timestep = self.resnet_model(batch_timestep)
                predictions_timestep = predictions_timestep.squeeze(1)
                predictions[timestep] = predictions_timestep

            # Predictions tensor with shape (batch_size, forecast horizon)
            predictions = torch.swapaxes(predictions, 0, 1)

        # Predict wind power generation for all time steps at once
        elif self.forecast_version == "all":
            predictions = self.resnet_model(batch)

        predictions = predictions.to(self.device)

        return predictions

    def configure_optimizers(self):
        """Configure the optimizers and learning rate schedulers.

        Returns:
            A tuple containing a list of optimizers and a list of scheduler dictionaries.
            The optimizer list contains a single Adam optimizer.
            The scheduler dictionary contains the ReduceLROnPlateau scheduler with 'val_loss' as the monitor.
        """
        # Use the Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Use the ReduceLROnPlateau learning rate scheduler
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min"
            ),
            "monitor": "val_loss",
        }

        return [optimizer], [scheduler]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a training step on the given batch of data.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the inputs and targets.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The calculated loss.

        """
        inputs, targets = batch

        # Start adversarial training
        if random.random() < self.p_adv_training:
            # Construct the PGD attacker
            pgd = PGD(self, self.criterion_adv_training)

            # Size of perturbation
            eps = self.eps_adv_training

            # Input requires gradients
            inputs.requires_grad_()

            # Create adversarial example
            inputs_adv = pgd.perturb(
                inputs=inputs,
                radius=eps,
                step_size=2 * eps / self.step_num_adv_training,
                step_num=self.step_num_adv_training,
                target=targets,
                targeted=False,
                norm=self.norm_adv_training,
            )

            inputs = inputs_adv

        predictions = self(inputs)

        # Calculate MSE loss
        loss = self.criterion(predictions, targets)

        # Calculate RMSE loss
        rmse = torch.sqrt(loss)

        # Perform logging
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_rmse", rmse, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a validation step on the given batch of data.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the inputs and targets.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The calculated loss.

        """
        inputs, targets = batch

        predictions = self(inputs)

        # Calculate MSE loss
        loss = self.criterion(predictions, targets)

        # Calculate RMSE loss
        rmse = torch.sqrt(loss)

        # Perform logging
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a test step on the given batch of data.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the inputs and targets.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The calculated loss.

        """
        inputs, targets = batch

        predictions = self(inputs)

        # Calculate MSE loss
        loss = self.criterion(predictions, targets)

        # Calculate RMSE loss
        rmse = torch.sqrt(loss)

        # Perform logging
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_rmse", rmse, on_step=False, on_epoch=True)
