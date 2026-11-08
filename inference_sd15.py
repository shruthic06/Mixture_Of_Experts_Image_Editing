#!/usr/bin/env python
"""
Inference script for SD1.5 InstructPix2Pix models (MoE and Dense).

Automatically detects model type from checkpoint.

Usage:
    python inference_sd15_moe.py \
        --checkpoint /scratch/khasti/sd15-moe-pix2pix/run-xxx/best_checkpoint.pt \
        --input_image input.jpg \
        --prompt "make it sunset" \
        --output output.jpg
"""

import argparse
import torch
from PIL import Image

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

# Import MoE components from training script
from train_sd15_moe_pix2pix import MoEFeedForward, apply_moe_to_unet
import torch.nn as nn


def load_unet_from_checkpoint(
    checkpoint_path: str,
    pretrained_model: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda",
):
    """
    Load trained UNet from checkpoint.
    Automatically detects MoE vs Dense from checkpoint.
    """
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Detect model type
    model_type = checkpoint.get('model_type', None)
    
    # If not tagged, try to detect from state dict keys
    if model_type is None:
        state_dict_keys = list(checkpoint['unet_state_dict'].keys())
        has_moe_keys = any('expert' in k.lower() or 'gate.gate' in k.lower() for k in state_dict_keys)
        model_type = 'moe' if has_moe_keys else 'dense'
        print(f"Model type (auto-detected): {model_type}")
    else:
        print(f"Model type (from checkpoint): {model_type}")
    
    print(f"Loading base UNet from {pretrained_model}...")
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model, subfolder="unet"
    )
    
    # Convert to InstructPix2Pix (8 input channels)
    print("Converting to InstructPix2Pix (8 input channels)...")
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
    
    # Apply MoE if needed
    if model_type == 'moe':
        num_experts = 4  # Default
        print(f"Applying MoE with {num_experts} experts...")
        num_replaced = apply_moe_to_unet(unet, num_experts=num_experts)
        print(f"Replaced {num_replaced} FFN blocks")
    else:
        print("Using Dense model (no MoE)")
    
    # Load trained weights
    print("Loading trained weights...")
    unet.load_state_dict(checkpoint['unet_state_dict'])
    
    unet = unet.to(device)
    unet.eval()
    
    print(f"Model loaded!")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Global step: {checkpoint.get('global_step', 'N/A')}")
    print(f"  Best loss: {checkpoint.get('best_loss', 'N/A')}")
    
    return unet, model_type


@torch.no_grad()
def run_inference(
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    input_image: Image.Image,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    image_guidance_scale: float = 1.5,
    device: str = "cuda",
):
    """Run InstructPix2Pix inference."""
    
    # Preprocess input image
    resolution = 512
    input_image = input_image.convert('RGB')
    input_image = input_image.resize((resolution, 344), Image.BILINEAR)
    
    # Convert to tensor
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # Encode input image
    input_latents = vae.encode(input_tensor).latent_dist.sample() * vae.config.scaling_factor
    
    # Encode prompt
    tokens = tokenizer(
        prompt,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    
    text_embeddings = text_encoder(tokens)[0]
    
    # Unconditional embeddings for classifier-free guidance
    uncond_tokens = tokenizer(
        "",
        max_length=tokenizer.model_max_length,
        padding="max_length",
        return_tensors="pt",
    ).input_ids.to(device)
    uncond_embeddings = text_encoder(uncond_tokens)[0]
    
    # Combine for classifier-free guidance
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings, text_embeddings])
    
    # Start from noise
    latents = torch.randn_like(input_latents)
    
    # Set timesteps
    noise_scheduler.set_timesteps(num_inference_steps)
    
    # Denoising loop
    for t in noise_scheduler.timesteps:
        # Expand latents for classifier-free guidance (3x for InstructPix2Pix)
        latent_model_input = torch.cat([latents] * 3)
        
        # Concat with input image latents
        input_latents_expanded = torch.cat([
            torch.zeros_like(input_latents),  # unconditional
            torch.zeros_like(input_latents),  # text only
            input_latents,                     # full conditioning
        ])
        latent_model_input = torch.cat([latent_model_input, input_latents_expanded], dim=1)
        
        # Predict noise
        noise_pred = unet(latent_model_input, t, text_embeddings).sample
        
        # Classifier-free guidance
        noise_pred_uncond, noise_pred_text, noise_pred_full = noise_pred.chunk(3)
        
        noise_pred = (
            noise_pred_uncond
            + guidance_scale * (noise_pred_text - noise_pred_uncond)
            + image_guidance_scale * (noise_pred_full - noise_pred_text)
        )
        
        # Denoise step
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode latents
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample
    
    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype("uint8")
    
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint')
    parser.add_argument('--input_image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Edit instruction')
    parser.add_argument('--output', type=str, default='output.jpg',
                        help='Output image path')
    parser.add_argument('--pretrained_model', type=str,
                        default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--image_guidance_scale', type=float, default=1.5)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load models
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model, subfolder="text_encoder"
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae"
    ).to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",  # Matching official InstructPix2Pix
        prediction_type="epsilon",
        clip_sample=False,
    )
    
    # Load MoE or Dense UNet
    unet, model_type = load_unet_from_checkpoint(
        checkpoint_path=args.checkpoint,
        pretrained_model=args.pretrained_model,
        device=device,
    )
    
    # Load input image
    print(f"Loading input image: {args.input_image}")
    input_image = Image.open(args.input_image)
    
    # Run inference
    print(f"Running inference with prompt: '{args.prompt}'")
    output_image = run_inference(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        input_image=input_image,
        prompt=args.prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        device=device,
    )
    
    # Save output
    output_image.save(args.output)
    print(f"Saved output to: {args.output}")


if __name__ == '__main__':
    main()
