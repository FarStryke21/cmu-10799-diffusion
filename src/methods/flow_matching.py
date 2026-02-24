"""
Flow Matching Method (Optimal Transport / Straight Path)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional    

from .base import BaseMethod


class FlowMatching(BaseMethod):
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        sigma_min: float = 1e-5,
    ):
        super().__init__(model, device)
        self.sigma_min = sigma_min
        print(f"Initializing FlowMatching (Optimal Transport) with sigma_min={sigma_min}")

    def compute_loss(self, batch, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the Flow Matching loss with Classifier-Free Guidance training.
        """
        # 1. Unpack Batch (Handle Tuple)
        if isinstance(batch, (tuple, list)):
            x1, labels = batch
        else:
            x1 = batch
            labels = None
            
        x1 = x1.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
            
        B = x1.shape[0]
        
        # 2. Sample Time t ~ Uniform[0, 1]
        t = torch.rand((B,), device=self.device)
        
        # 3. Sample Noise x0 ~ N(0, I)
        x0 = torch.randn_like(x1)
        
        # 4. Compute Linear Interpolation (OT Path)
        # x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
        # Simplified (if sigma_min=0): x_t = (1-t)*x0 + t*x1
        t_view = t.view(B, *([1] * (len(x1.shape) - 1)))
        x_t = (1 - (1 - self.sigma_min) * t_view) * x0 + t_view * x1
        
        # 5. Compute Target Velocity
        # v_t = dx_t/dt = x1 - (1 - sigma_min) * x0
        target_v = x1 - (1 - self.sigma_min) * x0
        
        # 6. Model Prediction with CFG Dropout
        if labels is not None:
            # Drop labels with probability 0.1
            drop_prob = 0.1
            context_mask = torch.bernoulli(torch.zeros(B) + drop_prob).bool().to(self.device)
            
            # Predict velocity field
            model_output = self.model(x_t, t, y=labels, context_mask=context_mask)
            
        else:
            # Unconditional training fallback
            model_output = self.model(x_t, t)
            
        # 7. Loss (MSE)
        loss = F.mse_loss(model_output, target_v)
        
        return loss, {"loss": loss.item()}
    

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 50,  # Default to 50 steps (much faster than DDPM 1000)
        labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 4.0,  # Guidance Strength
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples using Euler ODE solver with Classifier-Free Guidance.
        """
        self.eval_mode()
        
        # 1. Initialize Noise x0
        x = torch.randn((batch_size, *image_shape), device=self.device)
        
        # 2. Handle Labels
        if labels is None:
            # Random attributes if none provided
            labels = torch.randint(0, 2, (batch_size, 40)).float().to(self.device)
        else:
            labels = labels.to(self.device)
            
        # 3. Time Steps (0 to 1)
        # We use linspace for Euler integration
        t_seq = torch.linspace(0, 1, num_steps + 1, device=self.device)
        dt = 1.0 / num_steps
        
        # 4. Euler Integration Loop
        for i in range(num_steps):
            t_current = t_seq[i]
            
            # Prepare Inputs for CFG (Batching Cond and Uncond together)
            # Input: [x, x]
            x_in = torch.cat([x, x])
            t_in = torch.full((batch_size * 2,), t_current.item(), device=self.device)
            
            # Labels: [labels, labels]
            y_in = torch.cat([labels, labels])
            
            # Mask: [False (Cond), True (Uncond)]
            mask_in = torch.cat([
                torch.zeros(batch_size, dtype=torch.bool, device=self.device), # Cond
                torch.ones(batch_size, dtype=torch.bool, device=self.device)   # Uncond
            ])
            
            # Predict Velocity
            v_pred_combined = self.model(x_in, t_in, y=y_in, context_mask=mask_in)
            
            # Split back
            v_cond, v_uncond = v_pred_combined.chunk(2)
            
            # CFG Formula for Velocity
            # v = v_uncond + w * (v_cond - v_uncond)
            v_prime = v_uncond + cfg_scale * (v_cond - v_uncond)
            
            # Euler Step: x_{t+1} = x_t + v * dt
            x = x + v_prime * dt
            
        return x