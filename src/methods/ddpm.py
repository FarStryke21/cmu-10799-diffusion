"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        # TODO: Add your own arguments here
        prediction_type: Literal["epsilon", "sample"] = "epsilon",
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.prediction_type = prediction_type

        print(f"Initializing DDPM with {num_timesteps} timesteps, prediction_type={prediction_type}")
        
        # Create betas schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)  # (T,)

        # Precompute alphas and their cumulative products
        alphas = 1.0 - betas  # (T,)
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # (T,)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # (T,)

        # Constant for the backwad process
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Samping noise variance
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )  # (T,)

        # Constants specifically for x0-prediction
        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # Coeff 2 (for x_t): sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    # =========================================================================
    # You can add, delete or modify as many functions as you would like
    # =========================================================================
    
    # Pro tips: If you have a lot of pseudo parameters that you will specify for each
    # model run but will be fixed once you specified them (say in your config),
    # then you can use super().register_buffer(...) for these parameters

    # Pro tips 2: If you need a specific broadcasting for your tensors,
    # it's a good idea to write a general helper function for that
    
    # =========================================================================
    # Forward process
    # =========================================================================

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Extract values from a 1-D tensor for a batch of indices t,
        and reshape to [batch_size, 1, 1, 1, ...] for broadcasting.

        Args:
            a: 1-D tensor to extract from (e.g., alphas_cumprod)
            t: Tensor of indices (batch_size,)
            x_shape: Shape of the target tensor for broadcasting

        Returns:
            extracted: Tensor of shape [batch_size, 1, 1, 1, ...]
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t).reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor, noise = None) -> torch.Tensor: # TODO: Add your own arguments here
        # TODO: Implement the forward (noise adding) process of DDPM
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t
    
    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x0: torch.Tensor, t: torch.Tensor = None, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        TODO: Implement your DDPM loss function here

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """
        # If the training loop only passes 'x0', we must sample 't' ourselves.
        if t is None:
            B = x0.shape[0]
            # Sample uniform random timesteps from 0 to T-1
            t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
            
        # Generate random noise (epsilon)
        noise = torch.randn_like(x0)
        
        # Forward Process (Diffusion)
        x_t = self.forward_process(x0, t, noise)
        
        # Neural Network Prediction
        model_output = self.model(x_t, t)
        
        # Switch target based on prediction_type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = x0
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        loss = F.mse_loss(model_output, target)
        return loss, {"loss": loss.item()}
        

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================
    
    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement one step of the DDPM reverse process

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments
        
        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """
        model_output = self.model(x_t, t)
        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        
        if self.prediction_type == "epsilon":
            # Standard DDPM (Predicting Noise)
            sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
            beta_t = self._extract(self.betas, t, x_t.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            
            # Formula: (x_t - beta/sqrt(1-alpha_bar) * eps) / sqrt(alpha)
            model_mean = sqrt_recip_alphas_t * (
                x_t - (beta_t / sqrt_one_minus_alphas_cumprod_t) * model_output
            )
            
        elif self.prediction_type == "sample":
            # Alternative DDPM (Predicting x0)
            pred_x0 = model_output
            # Optional: Clip for stability (helps significantly)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
            coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
            
            # Formula: coef1 * x0 + coef2 * x_t
            model_mean = coef1 * pred_x0 + coef2 * x_t
            
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        noise = torch.randn_like(x_t)
        mask = (t > 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        x_prev = model_mean + mask * torch.sqrt(posterior_variance_t) * noise
        return x_prev

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        One step of DDIM.
        
        Args:
            x_t: Current sample
            t: Current timestep (integer)
            t_prev: Next timestep (integer)
            eta: Weight of noise (0.0 = Deterministic DDIM, 1.0 = DDPM)
        """
        # Get model prediction
        t_tensor = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
        model_output = self.model(x_t, t_tensor)
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t_tensor, x_t.shape)
        
        if t_prev >= 0:
            t_prev_tensor = torch.full((x_t.shape[0],), t_prev, device=self.device, dtype=torch.long)
            alpha_cumprod_prev = self._extract(self.alphas_cumprod, t_prev_tensor, x_t.shape)
        else:
            alpha_cumprod_prev = torch.ones_like(alpha_cumprod_t)

        # formula: x0 = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        
        if self.prediction_type == "epsilon":
            epsilon = model_output
            pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * epsilon) / sqrt_alpha_cumprod_t
        else: # "sample"
            pred_x0 = model_output
            epsilon = (x_t - sqrt_alpha_cumprod_t * pred_x0) / sqrt_one_minus_alpha_cumprod_t

        # Clip x0 for stability
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        sigma_t = eta * torch.sqrt(
            (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev)
        )

        # Direction pointing to x_t -> dir = sqrt(1 - alpha_prev - sigma^2) * epsilon
        dir_xt = torch.sqrt(1.0 - alpha_cumprod_prev - sigma_t**2) * epsilon

        noise = torch.randn_like(x_t)
        x_prev = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + sigma_t * noise
        
        return x_prev


    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        sampler: str = "ddpm",  # <--- NEW ARGUMENT
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples using either DDPM or DDIM.
        """
        self.eval_mode()
        x = torch.randn((batch_size, *image_shape), device=self.device)
        
        if num_steps is None:
            num_steps = self.num_timesteps

        if sampler == "ddim":
            skip = self.num_timesteps // num_steps
            time_seq = list(range(0, self.num_timesteps, skip))
            
            time_seq = list(reversed(time_seq))
            
            for i, t in enumerate(time_seq):
                if i < len(time_seq) - 1:
                    t_prev = time_seq[i+1]
                else:
                    t_prev = -1

                x = self.ddim_step(x, t, t_prev, eta=0.0)
                
        else:
            # Standard DDPM 
            for t in reversed(range(self.num_timesteps)):
                t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                x = self.reverse_process(x, t_batch)

        return x

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        # TODO: add other things you want to save
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            # TODO: add your parameters here
            prediction_type=ddpm_config.get("prediction_type", "epsilon"),
        )
