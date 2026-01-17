import os
import sys

# 1. Mac OpenMP Fix (Must be before torch import)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from src.models.unet import UNet
from src.methods.ddpm import DDPM

def test_integration():
    print("==================================================================")
    print("🚀 TEST 1: INTEGRATION (Does it compile and run?)")
    print("==================================================================")
    
    device = torch.device("cpu") # Keep it simple for debugging

    # A. Setup Tiny Model
    # We use small channels (32) so this runs instantly on CPU
    print("Step 1: Initializing UNet...")
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32, 
        channel_mult=(1, 2),    # Very shallow
        num_res_blocks=1,
        attention_resolutions=[16], # Matches 32x32 input at downsample 2
        num_heads=2
    ).to(device)
    
    # B. Setup DDPM
    print("Step 2: Initializing DDPM...")
    ddpm = DDPM(
        model=model,
        device=device,
        num_timesteps=100, # Short schedule for test
        beta_start=0.0001,
        beta_end=0.02
    )

    # C. Create Dummy Data
    # Batch size 2, 32x32 image
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32).to(device)
    t = torch.randint(0, 100, (batch_size,)).to(device)

    # D. Test Forward Pass
    print("Step 3: Running Forward Pass (Compute Loss)...")
    try:
        loss, stats = ddpm.compute_loss(x, t)
        print(f"   ✅ Loss Computed: {loss.item():.4f}")
    except Exception as e:
        print(f"   ❌ Forward Pass Failed: {e}")
        raise e

    # E. Test Backward Pass
    print("Step 4: Running Backward Pass (Gradient Check)...")
    try:
        loss.backward()
        print("   ✅ Gradients Computed Successfully")
    except Exception as e:
        print(f"   ❌ Backward Pass Failed: {e}")
        raise e

    # F. Test Sampling
    print("Step 5: Testing Sampling Loop...")
    try:
        # Sample just 1 step to verify the loop structure works
        # (We override internal num_timesteps logic for speed if needed, 
        # but here we just run the small 100 steps)
        samples = ddpm.sample(batch_size=2, image_shape=(3, 32, 32))
        print(f"   ✅ Sampling Successful! Output shape: {samples.shape}")
    except Exception as e:
        print(f"   ❌ Sampling Failed: {e}")
        raise e

    return ddpm, model, x

def test_overfitting(ddpm, model, batch):
    print("\n==================================================================")
    print("📉 TEST 2: OVERFITTING (Can it memorize one batch?)")
    print("==================================================================")
    
    # Use a standard optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Fix the time steps for overfitting so the target is constant
    # (Makes it easier to see loss drop)
    fixed_t = torch.randint(0, 100, (batch.shape[0],))
    
    print(f"Initial Loss check...")
    initial_loss, _ = ddpm.compute_loss(batch, fixed_t)
    print(f"Starting Loss: {initial_loss.item():.4f}")
    
    print("Training loop (50 iterations)...")
    
    losses = []
    for i in range(50):
        optimizer.zero_grad()
        loss, _ = ddpm.compute_loss(batch, fixed_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (i+1) % 10 == 0:
            print(f"Iter {i+1} | Loss: {loss.item():.5f}")

    if losses[-1] < losses[0] * 0.5:
        print("✅ SUCCESS: Loss dropped significantly!")
    else:
        print("⚠️ WARNING: Loss didn't drop much. Check your learning rate or architecture.")

if __name__ == "__main__":
    # Run Integration
    ddpm_instance, model_instance, dummy_data = test_integration()
    
    # Run Overfit
    test_overfitting(ddpm_instance, model_instance, dummy_data)