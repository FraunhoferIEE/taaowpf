import random
import sys
from typing import Tuple

from captum.robust import PGD
import pytorch_lightning as pl
import torch
from torch import nn

sys.path.append("/home/rheinrich/taaowpf")
from robustness_evaluation.robustness_scores import MSELossBatch


class EncoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        """
        Initializes an LSTM encoder module (encodes time-series sequence).

        Args:
            input_size (int): The number of features in the input X.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers (i.e., 2 means there are 2 stacked LSTMs).
        """

        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, input_batch: torch.Tensor):
        """
        Performs forward pass through the LSTM encoder.

        Args:
            input_batch: Input of shape (seq_len, # in batch, input_size).

        Returns:
            lstm_out: All the hidden states in the sequence.
            hidden: The hidden state and cell state for the last element in the sequence.
        """

        lstm_out, self.hidden = self.lstm(input_batch)

        return lstm_out, self.hidden


class DecoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        """
        Decoder LSTM module (decodes hidden state output by encoder)

        Args:
            input_size (int): The number of features in the input X.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers (i.e., 2 means there are 2 stacked LSTMs).
        """

        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Flatten(), nn.Linear(hidden_size, 1), nn.LeakyReLU()
        )

    def forward(self, input_batch: torch.Tensor, encoder_hidden_states: tuple):
        """
        Forward pass of the decoder LSTM module.

        Args:
            input_batch (torch.Tensor): Input tensor of shape (batch_size, input_size).
            encoder_hidden_states (tuple): Hidden states of the encoder LSTM.

        Returns:
            tuple: Output tensor giving all the hidden states in the sequence, and
                   a tuple containing the hidden state and cell state for the last
                   element in the sequence.
        """

        lstm_out, self.hidden = self.lstm(input_batch, encoder_hidden_states)
        output = self.out(lstm_out)
        return output, self.hidden


class WPF_AutoencoderLSTM(pl.LightningModule):
    def __init__(
        self,
        forecast_horizon: int,
        n_past_timesteps: int,
        hidden_size: int,
        num_layers: int,
        learning_rate: float = 1e-3,
        p_adv_training: float = 0.0,
        eps_adv_training: float = 0.15,
        step_num_adv_training: int = 100,
        norm_adv_training: str = "Linf",
    ):
        """
        WPF_AutoencoderLSTM model used to forecast wind power generation for individual wind farms.

        Args:
            forecast_horizon (int): Number of time steps to forecast.
            n_past_timesteps (int): Number of past timesteps considered for prediction.
            hidden_size (int): Size of the hidden state in LSTM.
            num_layers (int): Number of LSTM layers.
            learning_rate (float): Learning rate for optimization. Defaults to 1e-3.
            p_adv_training (float): Probability that input will be perturbed by adversarial attacks during training. Defaults to 0.0.
            eps_adv_training (float): Maximum perturbation caused by adversarial attacks. Defaults to 0.15.
            step_num_adv_training (int): Number of PGD iterations for adversarial attacks. Defaults to 100.
            norm_adv_training (str): Norm used to calculate adversarial attacks. Defaults to 'Linf'.
        """
        super(WPF_AutoencoderLSTM, self).__init__()

        # Forecast horizon
        self.forecast_horizon = forecast_horizon

        # Number of past timesteps considered for prediction
        self.n_past_timesteps = n_past_timesteps

        # Encoder only gets past wind power as input, so input size is 1
        self.encoder = EncoderLSTM(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers
        )

        # Decoder gets wind speed and wind power forecasts as inputs, so input size is 2
        self.decoder = DecoderLSTM(
            input_size=2, hidden_size=hidden_size, num_layers=num_layers
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

    def forward(self, inputs_windspeed: torch.Tensor, inputs_windpower: torch.Tensor):
        """
        Forward pass of the WPF_AutoencoderLSTM model.

        Args:
            inputs_windspeed (torch.Tensor): Input tensor of wind speed forecasts.
            inputs_windpower (torch.Tensor): Input tensor of past wind power measurements.

        Returns:
            torch.Tensor: Tensor of wind power predictions.

        """

        # Encoder outputs
        encoder_output, encoder_hidden = self.encoder(inputs_windpower)

        # Initialize hidden state of decoder with hidden state of encoder
        decoder_hidden = encoder_hidden

        # Get the last past wind power measurement
        decoder_input = inputs_windpower[:, -1, :]  # shape: (batch_size, input_size)

        # Use the last past wind power measurement together with the wind speed forecasts for the first prediction time step as input for the decoder
        decoder_input = torch.cat(
            (inputs_windspeed[:, 0, :], decoder_input), dim=1
        )  # shape: (batch_size, input_size)

        # Predictions tensor filled with zeros and shape (forecast horizon, batch_size)
        predictions = torch.zeros(self.forecast_horizon, inputs_windspeed.shape[0])

        # Step-by-step prediction of wind power generation for different time steps
        for timestep in range(self.forecast_horizon):
            decoder_input = decoder_input.unsqueeze(
                1
            )  # shape: (batch_size, timesteps, input_size)

            # Select the channels (timesteps) that are used to predict the wind power for this specific time step
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            predictions[timestep] = decoder_output.squeeze(1)

            if timestep < (self.forecast_horizon - 1):
                decoder_input = torch.cat(
                    (inputs_windspeed[:, timestep + 1, :], decoder_output), dim=1
                )

        # Predictions tensor with shape (batch_size, forecast horizon)
        predictions = torch.swapaxes(predictions, 0, 1)
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

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """
        Training step of the WPF_AutoencoderLSTM model.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing input tensors of wind speed forecasts,
                past wind power measurements, and target wind power values.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The calculated loss.

        """
        inputs_windspeed, inputs_windpower, targets = batch

        # Adversarial training
        if random.random() < self.p_adv_training:
            # Construct the PGD attacker
            pgd = PGD(self, self.criterion_adv_training)

            # Size of perturbation
            eps = self.eps_adv_training

            # Input requires gradients
            inputs_windspeed.requires_grad_()

            # Create adversarial example
            inputs_windspeed_adv = pgd.perturb(
                inputs=inputs_windspeed,
                radius=eps,
                step_size=2 * eps / self.step_num_adv_training,
                step_num=self.step_num_adv_training,
                target=targets,
                targeted=False,
                norm=self.norm_adv_training,
                additional_forward_args=inputs_windpower,
            )

            inputs_windspeed = inputs_windspeed_adv

        predictions = self(inputs_windspeed, inputs_windpower)

        # Calculate MSE loss
        loss = self.criterion(predictions, targets)

        # Calculate RMSE loss
        rmse = torch.sqrt(loss)

        # Perform logging
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_rmse", rmse, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """
        Validation step of the WPF_AutoencoderLSTM model.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing input tensors of wind speed forecasts,
                past wind power measurements, and target wind power values.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The calculated loss.

        """
        inputs_windspeed, inputs_windpower, targets = batch

        predictions = self(inputs_windspeed, inputs_windpower)

        # Calculate MSE loss
        loss = self.criterion(predictions, targets)

        # Calculate RMSE loss
        rmse = torch.sqrt(loss)

        # Perform logging
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """
        Test step of the WPF_AutoencoderLSTM model.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing input tensors of wind speed forecasts,
                past wind power measurements, and target wind power values.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The calculated loss.

        """
        inputs_windspeed, inputs_windpower, targets = batch

        predictions = self(inputs_windspeed, inputs_windpower)

        # Calculate MSE loss
        loss = self.criterion(predictions, targets)

        # Calculate RMSE loss
        rmse = torch.sqrt(loss)

        # Perform logging
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_rmse", rmse, on_step=False, on_epoch=True)
