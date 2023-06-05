from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    def __init__(self):
        """Root Mean Squared Error (RMSE) Loss.

        This loss calculates the root mean squared error between the predictions
        and the target.

        """
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        """Calculates the RMSE loss between the predictions and the target.

        Args:
            prediction (torch.Tensor): The predicted values.
            target (torch.Tensor): The target values.

        Returns:
            torch.Tensor: RMSE loss value.
        """
        return torch.sqrt(self.mse(prediction, target))


class BIASLoss(nn.Module):
    def __init__(self):
        """BIAS Loss.

        This loss calculates the bias between the predictions and the target.

        """
        super(BIASLoss, self).__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        """Calculates the BIAS loss between the predictions and the target.

        The BIAS loss is calculated as the mean deviation between the predictions and the target.

        Args:
            prediction (torch.Tensor): The predicted values.
            target (torch.Tensor): The target values.

        Returns:
            torch.Tensor: BIAS loss value.
        """
        deviation = prediction - target
        bias = deviation.mean()
        return bias


class MSELossBatch(nn.Module):
    def __init__(self):
        """Mean Squared Error (MSE) Loss for Batches.

        This loss calculates the mean squared error between the inputs and the targets for a batch.

        """
        super(MSELossBatch, self).__init__()

        self.mse = nn.MSELoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Calculates the MSE loss between the inputs and the targets for a batch.

        The MSE loss is calculated for each sample in the batch and then averaged.

        Args:
            inputs (torch.Tensor): The input values.
            targets (torch.Tensor): The target values.

        Returns:
            torch.Tensor: MSE loss value for the batch.
        """

        mse = self.mse(inputs, targets)
        mse = mse.mean(dim=1)

        return mse


class MSELossBatch_Area(nn.Module):
    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
    ):
        """Mean Squared Error (MSE) Loss for Batches with Area Limitation.

        This loss calculates the MSE between the inputs and the targets for a batch,
        but also considers the area between a lower and upper bound.

        Args:
            lower_bound (float): The lower bound for the area limitation.
            upper_bound (float): The upper bound for the area limitation.
        """
        super(MSELossBatch_Area, self).__init__()

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Calculates the MSE loss for the limited area between the inputs and the targets.

        The MSE loss is calculated for each sample in the batch, but considers the area
        between the lower and upper bound. Deviations inside this area are not considered.

        Args:
            inputs (torch.Tensor): The input values.
            targets (torch.Tensor): The target values.

        Returns:
            torch.Tensor: MSE loss value for the limited area.
        """

        deviation_lower = self.lower_bound - inputs

        deviation_upper = inputs - self.upper_bound

        deviation_area = F.relu(torch.maximum(deviation_lower, deviation_upper))

        mse_area = torch.square(deviation_area).mean(dim=1)

        return mse_area


class MSELossBatch_SemiTargeted(nn.Module):
    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        lambda_weight: float,
    ):
        """Mean Squared Error (MSE) Loss for Semi-Targeted Attacks.

        This loss calculates the MSE between the inputs and the targets for a batch,
        with an additional penalty term for deviations outside a specified area.

        Args:
            lower_bound (float): The lower bound for the area limitation.
            upper_bound (float): The upper bound for the area limitation.
            lambda_weight (float): The weight for the penalty term.
        """
        super(MSELossBatch_SemiTargeted, self).__init__()

        self.mse = MSELossBatch()
        self.mse_area = MSELossBatch_Area(
            lower_bound=lower_bound, upper_bound=upper_bound
        )

        self.lambda_weight = lambda_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Calculates the semi-targeted MSE loss for the batch.

        The MSE loss is calculated between the inputs and the targets, and an additional
        penalty term is added based on deviations outside the specified area between
        the lower and upper bounds.

        Args:
            inputs (torch.Tensor): The input values.
            targets (torch.Tensor): The target values.

        Returns:
            torch.Tensor: Semi-targeted MSE loss value.
        """

        mse = self.mse(inputs, targets)

        mse_area = self.mse_area(inputs, targets)

        mse_semi_targeted = mse - self.lambda_weight * mse_area

        return mse_semi_targeted


class RMSELossBatch(nn.Module):
    def __init__(self):
        """Root Mean Squared Error (RMSE) Loss for Batches.

        This loss calculates the RMSE between the predictions and the targets for a batch.

        """
        super().__init__()
        self.mse = MSELossBatch()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        """Calculates the RMSE loss for the batch.

        The RMSE loss is calculated by taking the square root of the mean squared error (MSE)
        between the predictions and the targets.

        Args:
            prediction (torch.Tensor): The predicted values.
            target (torch.Tensor): The target values.

        Returns:
            torch.Tensor: RMSE loss value.
        """
        return torch.sqrt(self.mse(prediction, target))


class RMSELossBatch_Area(nn.Module):
    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
    ):
        """Root Mean Squared Error (RMSE) Loss for Batches with Area Limitation.

        This loss calculates the RMSE between the inputs and the targets for a batch,
        but also considers the area between a lower and upper bound.

        Args:
            lower_bound (float): The lower bound of the area.
            upper_bound (float): The upper bound of the area.
        """
        super().__init__()

        self.mse_area = MSELossBatch_Area(
            lower_bound=lower_bound, upper_bound=upper_bound
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        """Calculates the RMSE loss for the limited area between the inputs and the targets.

        The RMSE loss is calculated for each sample in the batch, but considers the area
        between the lower and upper bound. Deviations inside this area are not considered.

        Args:
            prediction (torch.Tensor): The prediction values.
            target (torch.Tensor): The target values.

        Returns:
            torch.Tensor: RMSE loss value for the limited area.
        """
        return torch.sqrt(self.mse_area(prediction, target))


class DRS(nn.Module):
    def __init__(self, loss_func: Callable, epsilon: float = 1e-10):
        """Deformation Robustness Score (DRS).

        This module calculates the Deformation Robustness Score (DRS) based on a given loss function.

        Args:
            loss_func (Callable): Loss function used to calculate the loss.
            epsilon (float, optional): Small value to prevent division by zero. Defaults to 1e-10.
        """
        super().__init__()
        self.loss_func = loss_func
        self.epsilon = epsilon

    def forward(
        self,
        prediction_original: torch.Tensor,
        prediction_attacked: torch.Tensor,
        target_attacker: torch.Tensor,
    ):
        """Calculate the Deformation Robustness Score (DRS).

        The DRS is calculated based on the given loss function and the predictions of the original
        and attacked inputs along with the target values of the attacker.

        Args:
            prediction_original (torch.Tensor): Predictions of the original inputs.
            prediction_attacked (torch.Tensor): Predictions of the attacked inputs.
            target_attacker (torch.Tensor): Target values of the attacker.

        Returns:
            torch.Tensor: Deformation Robustness Score.
        """

        loss_original = self.loss_func(prediction_original, target_attacker)
        loss_attacked = self.loss_func(prediction_attacked, target_attacker)

        loss_ratio = loss_original / (loss_attacked + self.epsilon)
        drs = torch.exp(1 - loss_ratio)
        drs = torch.minimum(drs, torch.ones_like(drs))

        return drs


# Performance Robustness Score
class PRS(nn.Module):
    def __init__(self, loss_func: Callable, epsilon: float = 1e-10):
        super().__init__()
        """Performance Robustness Score (PRS).

        This module calculates the Performance Robustness Score (PRS) based on a given loss function.
                
        Args:
            loss_func (Callable): Loss function used to calculate the loss.
            epsilon (float, optional): Small value to prevent division by zero. Defaults to 1e-10.
        """
        self.loss_func = loss_func
        self.epsilon = epsilon

    def forward(
        self,
        prediction_original: torch.Tensor,
        prediction_attacked: torch.Tensor,
        target_original: torch.Tensor,
    ):
        """Calculate the Performance Robustness Score (PRS).

        The PRS is calculated based on the given loss function and the predictions of the original
        and attacked inputs along with the target values for the original inputs.

        Args:
            prediction_original (torch.Tensor): Predictions of the original inputs.
            prediction_attacked (torch.Tensor): Predictions of the attacked inputs.
            target_original (torch.Tensor): Target values for the original inputs.

        Returns:
            torch.Tensor: Performance Robustness Score.
        """

        loss_original = self.loss_func(prediction_original, target_original)
        loss_attacked = self.loss_func(prediction_attacked, target_original)

        loss_ratio = loss_attacked / (loss_original + self.epsilon)
        prs = torch.exp(1 - loss_ratio)
        prs = torch.minimum(prs, torch.ones_like(prs))

        return prs


# Targeted Attack Robustness Score
class TARS(nn.Module):
    def __init__(
        self,
        loss_func_drs: Callable,
        loss_func_prs: Callable,
        beta: float = 1.0,
        epsilon: float = 1e-10,
    ):
        """Targeted Attack Robustness Score (TARS).

        This module calculates the Targeted Attack Robustness Score (TARS) based on two given loss functions: one for
        calculating the Deformation Robustness Score (DRS) and one for calculating the Performance Robustness Score (PRS).

        Args:
            loss_func_drs (Callable): Loss function used to calculate the Deformation Robustness Score (DRS).
            loss_func_prs (Callable): Loss function used to calculate the Performance Robustness Score (PRS).
            beta (float, optional): Beta parameter. Defaults to 1.0.
            epsilon (float, optional): Small value to prevent division by zero. Defaults to 1e-10.
        """
        super().__init__()
        self.beta = beta
        self.drs = DRS(loss_func=loss_func_drs, epsilon=epsilon)
        self.prs = PRS(loss_func=loss_func_prs, epsilon=epsilon)

    def forward(
        self,
        prediction_original: torch.Tensor,
        prediction_attacked: torch.Tensor,
        target_original: torch.Tensor,
        target_attacker: torch.Tensor,
    ):
        """Calculate the Targeted Attack Robustness Score (TARS).

        The TARS is calculated based on the Deformation Robustness Score (DRS) and the Performance Robustness Score (PRS),
        both calculated using the given loss functions. It takes predictions of the original and attacked inputs,
        along with the target values for the original inputs and the target values of the attacker.

        Args:
            prediction_original (torch.Tensor): Predictions of the original inputs.
            prediction_attacked (torch.Tensor): Predictions of the attacked inputs.
            target_original (torch.Tensor): Target values for the original inputs.
            target_attacker (torch.Tensor): Target values of the attacker.

        Returns:
            torch.Tensor: Targeted Attack Robustness Score.
        """

        drs = self.drs(prediction_original, prediction_attacked, target_attacker)
        prs = self.prs(prediction_original, prediction_attacked, target_original)

        numerator = (1 + (self.beta**2)) * drs * prs
        denominator = ((self.beta**2) * prs) + drs
        tars = numerator / denominator

        return tars
