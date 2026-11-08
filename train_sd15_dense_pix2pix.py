#!/usr/bin/env python
"""
SD1.5 Dense UNet Training for InstructPix2Pix (Baseline)

Matches official InstructPix2Pix training implementation:
- 8-channel input with zero-initialized channel expansion
- Dual classifier-free guidance training (text + image dropout)
- Original latents use deterministic .mode() not .sample()
- Scaled linear beta schedule
- Learning rate 5e-5 (diffusers default) or 1e-4 (original paper)

Usage:
    torchrun --nproc_per_node=2 train_sd15_dense_pix2pix.py \
        --dataset_path /scratch/khasti/datasets/instructpix2pix \
        --output_dir /scratch/khasti/sd15-dense-pix2pix
"""

import argparse
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from PIL import Image
from datasets import load_from_disk
from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


# =============================================================================
# Dataset - Matching InstructPix2Pix
# =============================================================================

class InstructPix2PixDataset(Dataset):
    """
    Dataset for InstructPix2Pix training.
    Matches official implementation with random horizontal flip.
    """
    
    def __init__(
        self,
        dataset_path: str,
        tokenizer: CLIPTokenizer,
        resolution: int = 256,
        max_samples: Optional[int] = None,
        random_flip: bool = True,
    ):
        self.dataset = load_from_disk(dataset_path)
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.random_flip = random_flip
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        original_image = item['original_image']
        edited_image = item['edited_image']
        
        if isinstance(original_image, str):
            original_image = Image.open(original_image).convert('RGB')
        elif not isinstance(original_image, Image.Image):
            original_image = Image.fromarray(original_image).convert('RGB')
        
        if isinstance(edited_image, str):
            edited_image = Image.open(edited_image).convert('RGB')
        elif not isinstance(edited_image, Image.Image):
            edited_image = Image.fromarray(edited_image).convert('RGB')
        
        # Resize
        original_image = original_image.resize(
            (self.resolution, self.resolution), Image.BILINEAR
        )
        edited_image = edited_image.resize(
            (self.resolution, self.resolution), Image.BILINEAR
        )
        
        # Random horizontal flip - apply to BOTH images identically
        if self.random_flip and torch.rand(1).item() > 0.5:
            original_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)
            edited_image = edited_image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Convert to tensors and normalize
        original_image = transforms.ToTensor()(original_image)
        edited_image = transforms.ToTensor()(edited_image)
        original_image = self.normalize(original_image)
        edited_image = self.normalize(edited_image)
        
        instruction = item.get('edit_prompt', item.get('instruction', ''))
        
        tokens = self.tokenizer(
            instruction,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            'original_image': original_image,
            'edited_image': edited_image,
            'input_ids': tokens.input_ids.squeeze(0),
        }


# =============================================================================
# Training Utilities
# =============================================================================

def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0 if dist.is_initialized() else True


def print_main(msg):
    if is_main_process():
        print(msg)


# =============================================================================
# Main Training - Matching InstructPix2Pix
# =============================================================================

def main(args):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    print_main("=" * 70)
    print_main("SD1.5 Dense UNet Training for InstructPix2Pix (Baseline)")
    print_main("Matching official InstructPix2Pix implementation")
    print_main("=" * 70)
    print_main(f"World size: {world_size}")
    print_main(f"Learning rate: {args.learning_rate}")
    print_main(f"Conditioning dropout: {args.conditioning_dropout_prob}")
    print_main("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Models
    print_main("Loading models...")
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model, subfolder="text_encoder"
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae"
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet"
    )
    
    # Noise scheduler - MATCHING OFFICIAL (scaled_linear beta schedule)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        clip_sample=False,
    )
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Modify UNet for 8 input channels (zero-init new channels)
    print_main("Converting UNet for InstructPix2Pix (8 channels, zero-init)...")
    
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
    unet = unet.to(device)
    
    total_params = sum(p.numel() for p in unet.parameters())
    print_main(f"UNet params: {total_params:,}")
    
    if world_size > 1:
        unet = DDP(unet, device_ids=[local_rank])
    
    # Null embeddings for CFG training
    null_tokens = tokenizer(
        "", max_length=tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    
    with torch.no_grad():
        null_text_embedding = text_encoder(null_tokens)[0]
    
    # Dataset
    print_main("Loading dataset...")
    dataset = InstructPix2PixDataset(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        resolution=args.resolution,
        max_samples=args.max_samples,
        random_flip=True,
    )
    print_main(f"Dataset size: {len(dataset):,} samples")
    
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Optimizer - matching official
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    
    num_training_steps = args.epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps
    )
    
    # Resume
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    
    if args.resume_from and os.path.exists(args.resume_from):
        print_main(f"Resuming from: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        
        unet_state = checkpoint['unet_state_dict']
        if hasattr(unet, 'module'):
            unet.module.load_state_dict(unet_state)
        else:
            unet.load_state_dict(unet_state)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_loss = checkpoint.get('best_loss', float('inf'))
        print_main(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Training Loop
    print_main(f"\nStarting training...")
    print_main(f"Steps per epoch: {len(dataloader)}")
    print_main("-" * 70)
    
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision else None
    
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        unet.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            original_images = batch['original_image'].to(device)
            edited_images = batch['edited_image'].to(device)
            input_ids = batch['input_ids'].to(device)
            batch_size = original_images.shape[0]
            
            # Encode images - MATCHING OFFICIAL
            # Original: .mode() (deterministic), Edited: .sample() (stochastic)
            with torch.no_grad():
                original_latents = vae.encode(original_images).latent_dist.mode()
                original_latents = original_latents * vae.config.scaling_factor
                
                edited_latents = vae.encode(edited_images).latent_dist.sample()
                edited_latents = edited_latents * vae.config.scaling_factor
                
                encoder_hidden_states = text_encoder(input_ids)[0]
            
            # Dual CFG training - MATCHING OFFICIAL
            random_p = torch.rand(batch_size, device=device)
            
            # Text dropout
            prompt_mask = random_p < 2 * args.conditioning_dropout_prob
            prompt_mask = prompt_mask.reshape(batch_size, 1, 1)
            null_text_expanded = null_text_embedding.expand(batch_size, -1, -1)
            encoder_hidden_states = torch.where(
                prompt_mask, null_text_expanded, encoder_hidden_states
            )
            
            # Image dropout
            image_dropout_mask = (
                (random_p >= args.conditioning_dropout_prob) & 
                (random_p < 3 * args.conditioning_dropout_prob)
            )
            image_dropout_mask = image_dropout_mask.reshape(batch_size, 1, 1, 1)
            original_latents = torch.where(
                image_dropout_mask,
                torch.zeros_like(original_latents),
                original_latents
            )
            
            # Add noise
            noise = torch.randn_like(edited_latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=device
            ).long()
            
            noisy_latents = noise_scheduler.add_noise(edited_latents, noise, timesteps)
            latent_input = torch.cat([noisy_latents, original_latents], dim=1)
            
            # Forward pass
            optimizer.zero_grad()
            
            if args.mixed_precision:
                with torch.amp.autocast('cuda'):
                    noise_pred = unet(latent_input, timesteps, encoder_hidden_states).sample
                    loss = F.mse_loss(noise_pred, noise)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                noise_pred = unet(latent_input, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred, noise)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            global_step += 1
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if global_step % args.log_every == 0:
                print_main(
                    f"Epoch {epoch+1}/{args.epochs} | Step {global_step} | "
                    f"Loss: {loss.item():.4f}"
                )
            
            if global_step % args.save_every == 0 and is_main_process():
                save_checkpoint(
                    unet, optimizer, scheduler, epoch, global_step, best_loss,
                    os.path.join(args.output_dir, f'checkpoint_step_{global_step}.pt'),
                    model_type='dense'
                )
        
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - start_time
        
        print_main(
            f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        if is_main_process():
            save_checkpoint(
                unet, optimizer, scheduler, epoch, global_step, best_loss,
                os.path.join(args.output_dir, 'latest_checkpoint.pt'),
                model_type='dense'
            )
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    unet, optimizer, scheduler, epoch, global_step, best_loss,
                    os.path.join(args.output_dir, 'best_checkpoint.pt'),
                    model_type='dense'
                )
                print_main(f"  └── New best model saved!")
    
    print_main("\n" + "=" * 70)
    print_main(f"Training complete! Best loss: {best_loss:.4f}")
    print_main("=" * 70)
    
    cleanup_distributed()


def save_checkpoint(unet, optimizer, scheduler, epoch, global_step, best_loss, path, model_type='dense'):
    unet_state = unet.module.state_dict() if hasattr(unet, 'module') else unet.state_dict()
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'best_loss': best_loss,
        'unet_state_dict': unet_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_type': model_type,
    }, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--pretrained_model', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--resume_from', type=str, default=None)
    
    # Training - Matching official InstructPix2Pix
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--mixed_precision', action='store_true')
    
    # Conditioning dropout - CRITICAL for InstructPix2Pix
    parser.add_argument('--conditioning_dropout_prob', type=float, default=0.05)
    
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    
    args = parser.parse_args()
    main(args)

