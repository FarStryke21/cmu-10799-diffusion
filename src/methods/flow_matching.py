"""
Flow Matching Method (Optimal Transport / Straight Path)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict

from .base import BaseMethod


class FlowMatching(BaseMethod):
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ):
        super().__init__(model, device)

    def compute_loss(self, x_1: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the Flow Matching loss (Conditional Flow Matching).
        
        Args:
            x_1: Clean data samples (Target) [-1, 1]
            
        Returns:
            loss: MSE loss between predicted velocity and ground truth velocity
            metrics: dict with loss value
        """
        b = x_1.shape[0]
        
        # 1. Sample t uniformly from [0, 1]
        t = torch.rand((b,), device=self.device)
        
        # 2. Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)
        
        # 3. Compute x_t (Linear Interpolation)
        # x_t = (1 - t) * x_0 + t * x_1
        # Need to reshape t for broadcasting: (B,) -> (B, 1, 1, 1)
        t_view = t.view(-1, *([1] * (x_1.ndim - 1)))
        x_t = (1 - t_view) * x_0 + t_view * x_1
        
        # 4. Compute Ground Truth Velocity
        # v_t = d/dt (x_t) = x_1 - x_0
        target_velocity = x_1 - x_0
        
        # 5. Predict Velocity
        # Note: We scale t by 999 to match the input range expected by the UNet's 
        # embedding layer (which was designed for DDPM's 0-1000 range).
        v_pred = self.model(x_t, t * 999)
        
        # 6. Loss (MSE)
        loss = F.mse_loss(v_pred, target_velocity)
        
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 100,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples using Euler Integration (ODE Solver).
        Integrates from t=0 (Noise) to t=1 (Data).
        """
        self.eval_mode()
        
        # 1. Start from pure noise x_0
        x = torch.randn((batch_size, *image_shape), device=self.device)
        
        # 2. Define time steps (0 to 1)
        # using linspace ensures we hit exactly 0.0 and 1.0
        times = torch.linspace(0, 1, num_steps + 1, device=self.device)
        dt = 1.0 / num_steps
        
        # 3. Euler Integration Loop
        for i in range(num_steps):
            t = times[i]
            
            # Create batch of current time t
            t_batch = torch.ones((batch_size,), device=self.device) * t
            
            # Predict velocity
            # Again, scale t by 999 for the model
            v_pred = self.model(x, t_batch * 999)
            
            # Update x: x_{t+1} = x_t + v(x_t, t) * dt
            x = x + v_pred * dt
            
        return x