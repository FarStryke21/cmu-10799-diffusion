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
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
        
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

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

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
        noise_pred = self.model(x_t, t)
        
        # Compute Loss (MSE)
        loss = F.mse_loss(noise_pred, noise)
        
        return loss, {"loss": loss.item(), "noise_mse": loss.item()}
        

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
        predicted_noise = self.model(x_t, t)

        # Extract coefficients
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        beta_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)

        # mean = (1/sqrt(alpha)) * (x_t - (beta / sqrt(1-alpha_bar)) * eps)
        model_mean = sqrt_recip_alphas_t * (
            x_t - (beta_t / sqrt_one_minus_alphas_cumprod_t) * predicted_noise
        )

        noise = torch.randn_like(x_t)

        # No noise when t=0

        mask = (t > 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        x_prev = model_mean + mask * torch.sqrt(posterior_variance_t) * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        # TODO: add your arguments here
        **kwargs
    ) -> torch.Tensor:
        """
        TODO: Implement DDPM sampling loop: start from pure noise, iterate through all the time steps using reverse_process()

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments (e.g., num_steps)
        
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()

        x_t = torch.randn((batch_size, *image_shape)).to(self.device)
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, dtype=torch.long).to(self.device)
            x_t = self.reverse_process(x_t, t_batch)

        return x_t

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
        )
