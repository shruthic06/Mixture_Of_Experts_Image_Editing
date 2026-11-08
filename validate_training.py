#!/usr/bin/env python
"""
Comprehensive validation script for SD1.5 InstructPix2Pix training.

This script validates:
1. Dataset loading and preprocessing
2. Model architecture (8-channel UNet, MoE application)
3. Dual CFG dropout logic
4. Forward/backward pass
5. Checkpoint save/load
6. Inference pipeline

Run this BEFORE starting full training to catch issues early.

Usage:
    # Quick test (no dataset needed)
    python validate_training.py --test_mode quick
    
    # Full test with dataset
    python validate_training.py --test_mode full --dataset_path /scratch/khasti/datasets/instructpix2pix
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import tempfile
import os

# ============================================================================
# Test 1: Import validation
# ============================================================================
def test_imports():
    """Test that all required packages can be imported."""
    print("\n" + "="*60)
    print("TEST 1: Import validation")
    print("="*60)
    
    errors = []
    
    try:
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        print("  ✓ diffusers")
    except ImportError as e:
        errors.append(f"diffusers: {e}")
        print(f"  ✗ diffusers: {e}")
    
    try:
        from diffusers.models.attention import BasicTransformerBlock
        print("  ✓ diffusers.models.attention.BasicTransformerBlock")
    except ImportError as e:
        errors.append(f"BasicTransformerBlock: {e}")
        print(f"  ✗ BasicTransformerBlock: {e}")
    
    try:
        from transformers import CLIPTextModel, CLIPTokenizer
        print("  ✓ transformers (CLIP)")
    except ImportError as e:
        errors.append(f"transformers: {e}")
        print(f"  ✗ transformers: {e}")
    
    try:
        from datasets import load_from_disk
        print("  ✓ datasets")
    except ImportError as e:
        errors.append(f"datasets: {e}")
        print(f"  ✗ datasets: {e}")
    
    try:
        from torchvision import transforms
        print("  ✓ torchvision")
    except ImportError as e:
        errors.append(f"torchvision: {e}")
        print(f"  ✗ torchvision: {e}")
    
    if errors:
        print(f"\n  FAILED: {len(errors)} import errors")
        return False
    print("\n  PASSED: All imports successful")
    return True


# ============================================================================
# Test 2: MoE component validation
# ============================================================================
def test_moe_components():
    """Test MoE components work correctly."""
    print("\n" + "="*60)
    print("TEST 2: MoE component validation")
    print("="*60)
    
    # Import MoE components
    try:
        from train_sd15_moe_pix2pix import SimpleMoEGate, Expert, MoEFeedForward
        print("  ✓ MoE components imported")
    except ImportError as e:
        print(f"  ✗ Failed to import MoE components: {e}")
        print("    Make sure train_sd15_moe_pix2pix.py is in the current directory")
        return False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test MoEFeedForward
    dim = 320
    batch_size = 2
    seq_len = 64
    num_experts = 4
    
    try:
        moe_ff = MoEFeedForward(dim=dim, mult=4, dropout=0.0, num_experts=num_experts).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)
        out = moe_ff(x)
        
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
        print(f"  ✓ MoEFeedForward forward pass: input {x.shape} -> output {out.shape}")
        
        # Check aux_loss
        aux_loss = moe_ff.aux_loss
        assert isinstance(aux_loss, (float, torch.Tensor)), f"aux_loss type error: {type(aux_loss)}"
        print(f"  ✓ Auxiliary loss computed: {float(aux_loss):.4f}")
        
    except Exception as e:
        print(f"  ✗ MoEFeedForward test failed: {e}")
        return False
    
    # Test backward pass
    try:
        x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
        moe_ff.zero_grad()
        out = moe_ff(x)
        loss = out.sum()
        loss.backward()
        
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in moe_ff.parameters())
        assert has_grad, "No gradients computed"
        print("  ✓ Backward pass successful")
        
    except Exception as e:
        print(f"  ✗ Backward pass failed: {e}")
        return False
    
    print("\n  PASSED: MoE components working correctly")
    return True


# ============================================================================
# Test 3: UNet modification validation
# ============================================================================
def test_unet_modification():
    """Test UNet can be modified for InstructPix2Pix."""
    print("\n" + "="*60)
    print("TEST 3: UNet modification validation")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        from diffusers import UNet2DConditionModel
        from train_sd15_moe_pix2pix import apply_moe_to_unet, MoEFeedForward
        
        print("  Loading UNet (this may take a moment)...")
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="unet"
        )
        
        original_params = sum(p.numel() for p in unet.parameters())
        print(f"  Original UNet params: {original_params:,}")
        
        # Test 8-channel conversion
        print("  Converting to 8 input channels...")
        in_channels = 8
        out_channels = unet.conv_in.out_channels
        
        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=unet.conv_in.kernel_size,
                stride=unet.conv_in.stride,
                padding=unet.conv_in.padding,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            new_conv_in.bias.copy_(unet.conv_in.bias)
            unet.conv_in = new_conv_in
        
        unet.config['in_channels'] = in_channels
        print(f"  ✓ UNet input channels: {unet.conv_in.in_channels}")
        
        # Test MoE application
        print("  Applying MoE to UNet...")
        num_replaced = apply_moe_to_unet(unet, num_experts=4)
        print(f"  ✓ Replaced {num_replaced} FFN blocks with MoE")
        
        moe_params = sum(p.numel() for p in unet.parameters())
        print(f"  MoE UNet params: {moe_params:,} ({moe_params/original_params:.2f}x)")
        
        # Count MoE layers
        moe_count = sum(1 for m in unet.modules() if isinstance(m, MoEFeedForward))
        assert moe_count == num_replaced, f"MoE count mismatch: {moe_count} vs {num_replaced}"
        print(f"  ✓ MoE layers verified: {moe_count}")
        
        # Test forward pass
        print("  Testing forward pass with 8-channel input...")
        unet = unet.to(device)
        unet.eval()
        
        batch_size = 1
        latent_h = latent_w = 32  # 256 / 8
        
        latent_input = torch.randn(batch_size, 8, latent_h, latent_w, device=device)
        timestep = torch.tensor([500], device=device)
        encoder_hidden_states = torch.randn(batch_size, 77, 768, device=device)
        
        with torch.no_grad():
            output = unet(latent_input, timestep, encoder_hidden_states).sample
        
        expected_shape = (batch_size, 4, latent_h, latent_w)
        assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} vs {expected_shape}"
        print(f"  ✓ Forward pass successful: {latent_input.shape} -> {output.shape}")
        
    except Exception as e:
        print(f"  ✗ UNet modification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n  PASSED: UNet modification working correctly")
    return True


# ============================================================================
# Test 4: Dual CFG dropout logic
# ============================================================================
def test_cfg_dropout():
    """Test dual classifier-free guidance dropout logic."""
    print("\n" + "="*60)
    print("TEST 4: Dual CFG dropout logic validation")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conditioning_dropout_prob = 0.05
    
    # Simulate many samples
    num_samples = 10000
    random_p = torch.rand(num_samples, device=device)
    
    # Text dropout: p < 2 * prob
    text_dropped = (random_p < 2 * conditioning_dropout_prob).sum().item()
    
    # Image dropout: prob <= p < 3 * prob
    image_dropped = ((random_p >= conditioning_dropout_prob) & 
                     (random_p < 3 * conditioning_dropout_prob)).sum().item()
    
    # Both dropped: prob <= p < 2 * prob
    both_dropped = ((random_p >= conditioning_dropout_prob) & 
                    (random_p < 2 * conditioning_dropout_prob)).sum().item()
    
    # Neither dropped
    neither_dropped = (random_p >= 3 * conditioning_dropout_prob).sum().item()
    
    text_only_dropped = text_dropped - both_dropped
    image_only_dropped = image_dropped - both_dropped
    
    print(f"  Dropout analysis (n={num_samples}, prob={conditioning_dropout_prob}):")
    print(f"    Text dropped only:  {text_only_dropped:5d} ({100*text_only_dropped/num_samples:.1f}%) - expected ~5%")
    print(f"    Image dropped only: {image_only_dropped:5d} ({100*image_only_dropped/num_samples:.1f}%) - expected ~5%")
    print(f"    Both dropped:       {both_dropped:5d} ({100*both_dropped/num_samples:.1f}%) - expected ~5%")
    print(f"    Neither dropped:    {neither_dropped:5d} ({100*neither_dropped/num_samples:.1f}%) - expected ~85%")
    
    # Verify percentages are approximately correct
    tolerance = 0.02  # 2% tolerance
    
    checks = [
        (text_only_dropped / num_samples, 0.05, "text_only"),
        (image_only_dropped / num_samples, 0.05, "image_only"),
        (both_dropped / num_samples, 0.05, "both"),
        (neither_dropped / num_samples, 0.85, "neither"),
    ]
    
    all_passed = True
    for actual, expected, name in checks:
        if abs(actual - expected) > tolerance:
            print(f"  ✗ {name}: {actual:.3f} not close to {expected:.3f}")
            all_passed = False
        else:
            print(f"  ✓ {name}: {actual:.3f} ≈ {expected:.3f}")
    
    if all_passed:
        print("\n  PASSED: Dual CFG dropout logic correct")
    else:
        print("\n  FAILED: Dropout percentages incorrect")
    
    return all_passed


# ============================================================================
# Test 5: Normalize transform validation
# ============================================================================
def test_normalize_transform():
    """Test that Normalize transform works with RGB images."""
    print("\n" + "="*60)
    print("TEST 5: Normalize transform validation")
    print("="*60)
    
    from torchvision import transforms
    
    # Create a dummy RGB image
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    
    try:
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        tensor = transforms.ToTensor()(img)
        normalized = normalize(tensor)
        
        assert normalized.shape == (3, 256, 256), f"Shape error: {normalized.shape}"
        assert normalized.min() >= -1.0 and normalized.max() <= 1.0, f"Range error: [{normalized.min()}, {normalized.max()}]"
        print(f"  ✓ Normalize transform works: {tensor.shape} -> range [{normalized.min():.2f}, {normalized.max():.2f}]")
        
    except Exception as e:
        print(f"  ✗ Normalize transform failed: {e}")
        return False
    
    print("\n  PASSED: Normalize transform working correctly")
    return True


# ============================================================================
# Test 6: Dataset loading (optional)
# ============================================================================
def test_dataset_loading(dataset_path: str):
    """Test dataset can be loaded and preprocessed."""
    print("\n" + "="*60)
    print("TEST 6: Dataset loading validation")
    print("="*60)
    
    if not dataset_path or not os.path.exists(dataset_path):
        print(f"  SKIPPED: Dataset path not provided or doesn't exist: {dataset_path}")
        return True
    
    try:
        from datasets import load_from_disk
        from transformers import CLIPTokenizer
        
        print(f"  Loading dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        print(f"  ✓ Dataset loaded: {len(dataset)} samples")
        
        # Check required columns
        sample = dataset[0]
        required_cols = ['original_image', 'edited_image']
        prompt_cols = ['edit_prompt', 'instruction']
        
        for col in required_cols:
            if col not in sample:
                print(f"  ✗ Missing required column: {col}")
                return False
            print(f"  ✓ Found column: {col}")
        
        # Check for prompt column
        has_prompt = any(col in sample for col in prompt_cols)
        if not has_prompt:
            print(f"  ✗ Missing prompt column (need edit_prompt or instruction)")
            return False
        print(f"  ✓ Found prompt column")
        
        # Test tokenizer
        tokenizer = CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="tokenizer"
        )
        prompt = sample.get('edit_prompt', sample.get('instruction', ''))
        tokens = tokenizer(prompt, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        print(f"  ✓ Tokenization works: '{prompt[:50]}...' -> {tokens.input_ids.shape}")
        
    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n  PASSED: Dataset loading working correctly")
    return True


# ============================================================================
# Test 7: Checkpoint save/load
# ============================================================================
def test_checkpoint_save_load():
    """Test checkpoint can be saved and loaded."""
    print("\n" + "="*60)
    print("TEST 7: Checkpoint save/load validation")
    print("="*60)
    
    try:
        from train_sd15_moe_pix2pix import MoEFeedForward
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            MoEFeedForward(dim=20, num_experts=4),
            nn.Linear(20, 10),
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        checkpoint = {
            'epoch': 5,
            'global_step': 1000,
            'best_loss': 0.123,
            'unet_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'model_type': 'moe',
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Checkpoint saved to {checkpoint_path}")
        
        # Load checkpoint
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        assert loaded['epoch'] == 5
        assert loaded['global_step'] == 1000
        assert loaded['model_type'] == 'moe'
        print(f"  ✓ Checkpoint loaded successfully")
        print(f"    epoch: {loaded['epoch']}, step: {loaded['global_step']}, type: {loaded['model_type']}")
        
        # Load state dict
        model2 = nn.Sequential(
            nn.Linear(10, 20),
            MoEFeedForward(dim=20, num_experts=4),
            nn.Linear(20, 10),
        )
        model2.load_state_dict(loaded['unet_state_dict'])
        print(f"  ✓ State dict loaded into new model")
        
        # Cleanup
        os.unlink(checkpoint_path)
        
    except Exception as e:
        print(f"  ✗ Checkpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n  PASSED: Checkpoint save/load working correctly")
    return True


# ============================================================================
# Test 8: Full forward pass simulation
# ============================================================================
def test_full_forward_pass():
    """Test complete forward pass with all components."""
    print("\n" + "="*60)
    print("TEST 8: Full forward pass simulation")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
        from train_sd15_moe_pix2pix import apply_moe_to_unet, get_moe_aux_loss
        
        print("  Loading models (this may take a moment)...")
        
        tokenizer = CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="text_encoder"
        ).to(device).eval()
        vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="vae"
        ).to(device).eval()
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="unet"
        )
        
        # Noise scheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
            clip_sample=False,
        )
        print("  ✓ Models loaded")
        
        # Modify UNet
        in_channels = 8
        out_channels = unet.conv_in.out_channels
        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=unet.conv_in.kernel_size,
                stride=unet.conv_in.stride,
                padding=unet.conv_in.padding,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            new_conv_in.bias.copy_(unet.conv_in.bias)
            unet.conv_in = new_conv_in
        unet.config['in_channels'] = in_channels
        
        # Apply MoE
        num_replaced = apply_moe_to_unet(unet, num_experts=4)
        unet = unet.to(device)
        print(f"  ✓ UNet modified: 8 channels, {num_replaced} MoE layers")
        
        # Freeze VAE and text encoder
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        
        # Create dummy batch
        batch_size = 2
        resolution = 256
        
        # Dummy images (simulate dataset output)
        original_images = torch.randn(batch_size, 3, resolution, resolution, device=device)
        edited_images = torch.randn(batch_size, 3, resolution, resolution, device=device)
        
        # Tokenize prompt
        prompts = ["make it sunset", "add snow"]
        tokens = tokenizer(prompts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = tokens.input_ids.to(device)
        
        print("  Simulating training step...")
        
        # Encode images
        with torch.no_grad():
            original_latents = vae.encode(original_images).latent_dist.mode()
            original_latents = original_latents * vae.config.scaling_factor
            
            edited_latents = vae.encode(edited_images).latent_dist.sample()
            edited_latents = edited_latents * vae.config.scaling_factor
            
            encoder_hidden_states = text_encoder(input_ids)[0]
        
        print(f"  ✓ Images encoded: {original_images.shape} -> {original_latents.shape}")
        
        # Add noise
        noise = torch.randn_like(edited_latents)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(edited_latents, noise, timesteps)
        
        # Concatenate
        latent_input = torch.cat([noisy_latents, original_latents], dim=1)
        print(f"  ✓ Latent input prepared: {latent_input.shape}")
        
        # Forward pass
        unet.train()
        noise_pred = unet(latent_input, timesteps, encoder_hidden_states).sample
        print(f"  ✓ UNet forward pass: {noise_pred.shape}")
        
        # Compute loss
        main_loss = F.mse_loss(noise_pred, noise)
        aux_loss = get_moe_aux_loss(unet)
        total_loss = main_loss + 0.01 * aux_loss
        
        print(f"  ✓ Loss computed: main={main_loss.item():.4f}, aux={aux_loss.item():.4f}")
        
        # Backward pass
        total_loss.backward()
        
        grad_norm = sum(p.grad.norm().item() for p in unet.parameters() if p.grad is not None)
        print(f"  ✓ Backward pass: grad_norm={grad_norm:.2f}")
        
    except Exception as e:
        print(f"  ✗ Full forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n  PASSED: Full forward pass working correctly")
    return True


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, default='quick', choices=['quick', 'full'])
    parser.add_argument('--dataset_path', type=str, default=None)
    args = parser.parse_args()
    
    print("="*60)
    print("SD1.5 InstructPix2Pix Training Validation")
    print("="*60)
    print(f"Test mode: {args.test_mode}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("="*60)
    
    results = {}
    
    # Always run these tests
    results['imports'] = test_imports()
    results['normalize'] = test_normalize_transform()
    results['cfg_dropout'] = test_cfg_dropout()
    results['moe_components'] = test_moe_components()
    results['checkpoint'] = test_checkpoint_save_load()
    
    # Full tests require GPU and models
    if args.test_mode == 'full':
        if not torch.cuda.is_available():
            print("\nWARNING: Full tests require CUDA. Skipping...")
        else:
            results['unet_modification'] = test_unet_modification()
            results['full_forward'] = test_full_forward_pass()
            
            if args.dataset_path:
                results['dataset'] = test_dataset_loading(args.dataset_path)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("ALL TESTS PASSED! Ready to start training.")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED! Please fix issues before training.")
        sys.exit(1)


if __name__ == '__main__':
    main()

