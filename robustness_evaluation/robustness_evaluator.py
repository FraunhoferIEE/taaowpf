import sys
from typing import Callable

import pandas as pd
import torch

sys.path.append("/home/rheinrich/taaowpf")
from robustness_evaluation.robustness_scores import DRS, PRS, RMSELossBatch, TARS


class AdversarialRobustnessEvaluator:
    def __init__(
        self,
        forward_func: Callable,
        dataloader: torch.utils.data.dataloader.DataLoader,
        attack: Callable,
        attack_kwargs: dict,
        targeted: bool,
        loss_func_prs: Callable,
        loss_func_drs: Callable,
        target_attacker: torch.Tensor = None,
        additional_inputs: bool = False,
        requires_original_target: bool = False,
    ) -> None:
        """Evaluates adversarial robustness of a model.

        Args:
            forward_func (Callable or torch.nn.Module): This can either be an instance
                of pytorch model or any modification of a model's forward
                function.
            dataloader (DataLoader): The dataloader containing the input data.
            attack (Callable or perturbation): This can either be an instance
                of a Captum Perturbation / Attack
                or any other perturbation or attack function such
                as a torchvision transform.
            attack_kwargs (dict): Additional arguments to be provided to the given attack.
                This should be provided as a dictionary of keyword arguments.
            targeted (bool): Specifies whether the attack is targeted or not.
            loss_func_prs (Callable): Loss function for PRS calculation.
            loss_func_drs (Callable): Loss function for DRS calculation.
            target_attacker (Tensor, optional): Target of the attacker. Only required for targeted attacks.
            additional_inputs (bool): Specifies whether additional inputs are present or not.
            requires_original_target (bool): Specifies whether the original target is required or not.
        """

        self.additional_inputs = additional_inputs
        self.attack = attack
        self.attack_kwargs = attack_kwargs
        self.dataloader = dataloader
        self.forward_func = forward_func
        self.targeted = targeted
        self.target_attacker = target_attacker
        self.requires_original_target = requires_original_target
        self.loss_func_prs = loss_func_prs
        self.loss_func_drs = loss_func_drs

        self.rmse_unmodified_target_original = None
        self.rmse_unmodified_target_attacker = None
        self.rmse_attacked_target_original = None
        self.rmse_attacked_target_attacker = None
        self.drs_list = None
        self.prs_list = None
        self.tars_list = None

    def evaluate_attack(self):
        """Evaluates the attack and returns the results as a DataFrame.

        Returns:
            DataFrame: Results of the attack evaluation.

        """
        self.rmse_unmodified_target_original = []
        self.rmse_unmodified_target_attacker = []
        self.rmse_attacked_target_original = []
        self.rmse_attacked_target_attacker = []
        self.drs_list = []
        self.prs_list = []
        self.tars_list = []

        rmse = RMSELossBatch()
        drs = DRS(loss_func=self.loss_func_drs)
        prs = PRS(loss_func=self.loss_func_prs)
        tars = TARS(loss_func_prs=self.loss_func_prs, loss_func_drs=self.loss_func_drs)

        for batch in self.dataloader:
            if self.additional_inputs:
                inputs, inputs_additional, targets_original = batch
            else:
                inputs, targets_original = batch
                inputs_additional = None

            # calculate model prediction with unmodified inputs
            with torch.no_grad():
                if inputs_additional is not None:
                    prediction_original = self.forward_func(
                        *(inputs, inputs_additional)
                    )
                else:
                    prediction_original = self.forward_func(inputs)

            # generate perturbed inputs
            if self.requires_original_target:
                input_dict = {
                    "inputs": inputs,
                    "additional_forward_args": inputs_additional,
                    "target": targets_original,
                }
            elif self.targeted:
                batch_size = inputs.shape[0]
                target_attacker_batch = self.target_attacker.repeat([batch_size, 1])
                input_dict = {
                    "inputs": inputs,
                    "additional_forward_args": inputs_additional,
                    "target": target_attacker_batch,
                }
            else:
                input_dict = {
                    "inputs": inputs,
                    "additional_forward_args": inputs_additional,
                }

            input_dict.update(self.attack_kwargs)

            inputs_perturbed = self.attack.perturb(**input_dict)

            # calculate model prediction with perturbed inputs
            with torch.no_grad():
                if inputs_additional is not None:
                    prediction_attacked = self.forward_func(
                        *(inputs_perturbed, inputs_additional)
                    )
                else:
                    prediction_attacked = self.forward_func(inputs_perturbed)

            self.rmse_unmodified_target_original += list(
                rmse(prediction_original, targets_original).numpy()
            )
            self.rmse_attacked_target_original += list(
                rmse(prediction_attacked, targets_original).numpy()
            )
            self.prs_list += list(
                prs(prediction_original, prediction_attacked, targets_original).numpy()
            )

            results = pd.DataFrame(
                [
                    self.rmse_unmodified_target_original,
                    self.rmse_attacked_target_original,
                    self.prs_list,
                ]
            ).T
            results.columns = [
                "RMSE_Unmodified_Target_Original",
                "RMSE_Attacked_Target_Original",
                "PRS",
            ]

            if self.target_attacker is not None:
                batch_size = inputs.shape[0]
                target_attacker_batch = self.target_attacker.repeat([batch_size, 1])

                self.rmse_unmodified_target_attacker += list(
                    rmse(prediction_original, target_attacker_batch).numpy()
                )
                self.rmse_attacked_target_attacker += list(
                    rmse(prediction_attacked, target_attacker_batch).numpy()
                )
                self.drs_list += list(
                    drs(
                        prediction_original, prediction_attacked, target_attacker_batch
                    ).numpy()
                )
                self.tars_list += list(
                    tars(
                        prediction_original,
                        prediction_attacked,
                        targets_original,
                        target_attacker_batch,
                    ).numpy()
                )

                results_target_attacker = pd.DataFrame(
                    [
                        self.drs_list,
                        self.tars_list,
                        self.rmse_unmodified_target_attacker,
                        self.rmse_attacked_target_attacker,
                    ]
                ).T
                results_target_attacker.columns = [
                    "DRS",
                    "TARS",
                    "RMSE_Unmodified_Target_Attacker",
                    "RMSE_Attacked_Target_Attacker",
                ]

                results = pd.concat([results, results_target_attacker], axis=1)

        self.results = results

        return results

    def summary_table(self):
        """Calculates the summary table from the results.

        Returns:
            pd.Series: Summary table of the results.
        """
        summary_table = self.results.mean()
        return summary_table
