from typing import Any, Callable, Tuple, Union
import torch


class NoiseAttack():

    def __init__(
        self,
        forward_func: Callable,
        loss_func: Callable = None,
        lower_bound: float = float("-inf"),
        upper_bound: float = float("inf"),
    ) -> None:
        """Noise Attack. Perturbation gets sampled from a uniform distribution on the interval [-eps, eps].
        
        Args:
            forward_func (callable): The pytorch model for which the attack is computed.
            loss_func (callable, optional): Loss function. The loss function should take in outputs of the
                        model and labels, and return a loss tensor. The default loss function is negative log.
            lower_bound (float, optional): Lower bound of input values.
            upper_bound (float, optional): Upper bound of input values.
        """
        super().__init__()
        self.forward_func = forward_func
        self.loss_func = loss_func
        self.bound = lambda x: torch.clamp(x, min=lower_bound, max=upper_bound)

    def perturb(
        self,
        inputs: torch.Tensor,
        target: Any,
        radius: float,
        step_num: int,
        additional_forward_args: Any = None,
        norm: str = "Linf",
    ) -> torch.Tensor:
        """
        This method computes and returns the perturbed input for each input tensor.
        It supports both targeted and non-targeted attacks.

        Args:

            inputs (tensor): Input for which adversarial attack is computed.
            target (any): True labels of inputs if non-targeted attack is
                        desired. Target class of inputs if targeted attack
                        is desired. Target will be passed to the loss function
                        to compute loss, so the type needs to match the
                        argument type of the loss function.
            radius (float): Radius of the neighbor ball centered around inputs.
                        The perturbation should be within this range.
            step_num (int): Step numbers. It usually guarantees that the perturbation
                        can reach the border.
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. These arguments are provided to
                        forward_func in order following the arguments in inputs.
                        Default: None.
            norm (str, optional): Specifies the norm to calculate distance from
                        original inputs: 'Linf'|'L2'.
                        Default: 'Linf'.

        Returns:

            - **perturbed inputs** (*tensor* or tuple of *tensors*):
                        Perturbed input for each
                        input tensor. The perturbed inputs have the same shape and
                        dimensionality as the inputs.
                        If a single tensor is provided as inputs, a single tensor
                        is returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.
        """

        def _clip(inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
            diff = outputs - inputs
            if norm == "Linf":
                return inputs + torch.clamp(diff, -radius, radius)
            elif norm == "L2":
                return inputs + torch.renorm(diff, 2, 0, radius)
            else:
                raise AssertionError("Norm constraint must be L2 or Linf.")
                
        def _format_additional_forward_args(additional_forward_args: Any) -> Union[None, Tuple]:
            if additional_forward_args is not None and not isinstance(additional_forward_args, tuple):
                additional_forward_args = (additional_forward_args,)
            return additional_forward_args
                
        def _forward_with_loss() -> torch.Tensor:
            additional_inputs = _format_additional_forward_args(additional_forward_args)
            if additional_inputs is not None:
                outputs = self.forward_func(*(inputs, *additional_inputs))   
            else:
                outputs = self.forward_func(inputs)  
            return self.loss_func(outputs, target)

        original_inputs = inputs.clone()
        perturbed_inputs = inputs.clone()
        
        perturbed_inputs_best = torch.empty_like(original_inputs)
        loss_best = torch.zeros(original_inputs.shape[0])

        for _i in range(step_num):
            perturbation = torch.randn(size = original_inputs.shape) * radius
            perturbed_inputs += perturbation
            
            perturbed_inputs = _clip(original_inputs, perturbed_inputs)

            perturbed_inputs = self.bound(perturbed_inputs).detach()
            
            loss = _forward_with_loss()
            
            loss_increase = (loss > loss_best)
        
            loss_best[loss_increase] = loss[loss_increase]

            perturbed_inputs_best[loss_increase] = perturbed_inputs[loss_increase]
            
            
        return perturbed_inputs_best