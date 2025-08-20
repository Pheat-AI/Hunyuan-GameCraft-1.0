"""
Fast sampling using pre-loaded models from the warm function.
Following the pattern from sample_batch.py
"""

import os
import torch
import tempfile
import shutil
import random
from pathlib import Path
from PIL import Image
from loguru import logger
from hymm_sp.data_kits.data_tools import save_videos_grid


def generate_video(input_data, model_warmer):
    """
    Generate a video using the pre-loaded models.
    
    Args:
        input_data: Dict containing:
            - image_path: Path to the reference image
            - output_dir: Directory to save results
            - prompt: Text prompt
            - negative_prompt: Negative prompt
            - actions: List of actions
            - size: (height, width) tuple
            - seed: Random seed
            - guidance_scale: CFG scale
            - infer_steps: Number of inference steps
            - use_fp8: Whether to use FP8
            - cpu_offload: Whether to use CPU offloading
        model_warmer: Pre-loaded model warmer instance
        
    Returns:
        dict: Contains generated video path and metadata
    """
    logger.info("🎬 Starting video generation with pre-loaded models...")
    
    # Get pre-loaded components
    hunyuan_video_sampler = model_warmer.get_sampler()
    ref_image_transform = model_warmer.get_image_transform()
    device = model_warmer.get_device()
    args = model_warmer.get_args()
    
    # Extract parameters from input_data
    img_path = Path(input_data["image_path"])
    output_dir = Path(input_data["output_dir"])
    prompt = input_data["prompt"]
    negative_prompt = input_data.get("negative_prompt", "")
    actions = input_data["actions"]
    H, W = input_data["size"]
    seed = input_data.get("seed", random.randint(0, 1_000_000))
    guidance_scale = input_data.get("guidance_scale", 6.0)
    infer_steps = input_data.get("infer_steps", 50)
    cpu_offload = input_data.get("cpu_offload", False)
    
    # Process action data
    action_list = [a.id for a in actions]
    action_speed_list = [a.speed for a in actions]
    total_frames = sum(a.frames for a in actions)
    
    # Verify image exists
    if not img_path.exists():
        raise RuntimeError(f"Reference image not found at: {img_path}")
    
    logger.info(f"Using reference image: {img_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load and preprocess reference image (following sample_batch.py pattern)
    raw_ref_images = [Image.open(img_path).convert('RGB')]
    
    # Apply transformations and prepare tensor for model input
    ref_images_pixel_values = [ref_image_transform(ref_image) for ref_image in raw_ref_images]
    ref_images_pixel_values = torch.cat(ref_images_pixel_values).unsqueeze(0).unsqueeze(2).to(device)
    
    logger.info("🔄 Encoding reference image to latent space...")
    
    # Encode reference images to latent space using VAE
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        if cpu_offload:
            # Move VAE components to GPU temporarily for encoding
            hunyuan_video_sampler.vae.quant_conv.to('cuda')
            hunyuan_video_sampler.vae.encoder.to('cuda')
        
        # Enable tiling for VAE to handle large images efficiently
        hunyuan_video_sampler.pipeline.vae.enable_tiling()
        
        # Encode image to latents and scale by VAE's scaling factor
        raw_last_latents = hunyuan_video_sampler.vae.encode(
            ref_images_pixel_values
        ).latent_dist.sample().to(dtype=torch.float16)
        raw_last_latents.mul_(hunyuan_video_sampler.vae.config.scaling_factor)
        raw_ref_latents = raw_last_latents.clone()
        
        # Clean up
        hunyuan_video_sampler.pipeline.vae.disable_tiling()
        if cpu_offload:
            # Move VAE components back to CPU after encoding
            hunyuan_video_sampler.vae.quant_conv.to('cpu')
            hunyuan_video_sampler.vae.encoder.to('cpu')
    
    logger.info("🎯 Generating video segments...")
    
    # Store references for generation loop
    ref_images = raw_ref_images
    last_latents = raw_last_latents
    ref_latents = raw_ref_latents
    out_cat = None
    
    # Generate video segments for each action
    for idx, action in enumerate(actions):
        # Determine if this is the first action and using image start
        is_image = (idx == 0)  # First action starts from image
        
        logger.info(f"🎬 Generating segment {idx+1}/{len(actions)} with action: {action.id}")
        
        # Generate video segment with the current action
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt,
            action_id=action.id,
            action_speed=action.speed,                    
            is_image=is_image,
            size=(704, 1216),  # Fixed size as per sample_batch.py
            seed=seed,
            last_latents=last_latents,
            ref_latents=ref_latents,
            video_length=action.frames,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            negative_prompt=negative_prompt,
            infer_steps=infer_steps,
            flow_shift=5.0,  # Default from sample_batch.py
            use_linear_quadratic_schedule=False,
            linear_schedule_end=0.1,
            use_deepcache=True,
            cpu_offload=cpu_offload,
            ref_images=ref_images,
            output_dir=str(output_dir),
            return_latents=True,
            use_sage=False,
        )
        
        # Update latents for next iteration (maintain temporal consistency)
        ref_latents = outputs["ref_latents"]
        last_latents = outputs["last_latents"]
        
        # Save generated video segments
        sub_samples = outputs['samples'][0]
        
        # Initialize or concatenate video segments
        if idx == 0:
            out_cat = sub_samples
        else:
            # Append new segment to existing video
            out_cat = torch.cat([out_cat, sub_samples], dim=2)
    
    # Save final combined video
    logger.info("💾 Saving final video...")
    video_path = output_dir / "generated_video.mp4"
    save_videos_grid(out_cat, str(video_path), n_rows=1, fps=24)
    
    logger.info(f"✅ Video generation complete! Saved to: {video_path}")
    
    return {
        "video_path": video_path,
        "height": H,
        "width": W, 
        "total_frames": total_frames,
        "log": f"Successfully generated video with {len(actions)} actions, {total_frames} total frames"
    }