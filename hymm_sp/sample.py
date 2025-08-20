"""
Fast sampling using pre-loaded models from the warm function.
"""

import torch
import tempfile
import shutil
from pathlib import Path
from PIL import Image
from fal.toolkit import download_file


def generate_video(input_data, model_warmer):
    """
    Generate a video using the pre-loaded models.
    
    Args:
        input_data: Input data containing image_url, prompt, actions, etc.
        model_warmer: Pre-loaded model warmer instance
        
    Returns:
        dict: Contains generated video path and metadata
    """
    print("🎬 Starting video generation with pre-loaded models...")
    
    # Import here since we're using clone_repository
    from hymm_sp.data_kits.data_tools import save_videos_grid
    
    # Get pre-loaded components
    sampler = model_warmer.get_sampler()
    image_transform = model_warmer.get_image_transform()
    device = model_warmer.get_device()
    sampler_args = model_warmer.get_sampler_args()
    
    # Create temporary directory for this generation
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Download and preprocess input image
        print("📥 Downloading and preprocessing reference image...")
        img_path = tmp_path / "reference.png"
        downloaded_file = download_file(input_data.image_url, target_dir=str(tmp_path))
        
        # Copy or move the downloaded file to our expected location
        if Path(downloaded_file).name != "reference.png":
            shutil.copy2(downloaded_file, img_path)
        else:
            img_path = Path(downloaded_file)
        
        # Load and preprocess the reference image
        raw_ref_image = Image.open(img_path).convert('RGB')
        ref_image_tensor = image_transform(raw_ref_image)
        ref_images_pixel_values = ref_image_tensor.unsqueeze(0).unsqueeze(2).to(device)
        
        print("🔄 Encoding reference image to latent space...")
        
        # Encode reference image to latent space using VAE
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            if sampler_args.cpu_offload:
                # Move VAE components to GPU temporarily for encoding
                sampler.vae.quant_conv.to('cuda')
                sampler.vae.encoder.to('cuda')
            
            # Enable tiling for VAE to handle large images efficiently
            sampler.pipeline.vae.enable_tiling()
            
            # Encode image to latents and scale by VAE's scaling factor
            raw_last_latents = sampler.vae.encode(
                ref_images_pixel_values
            ).latent_dist.sample().to(dtype=torch.float16)
            raw_last_latents.mul_(sampler.vae.config.scaling_factor)
            raw_ref_latents = raw_last_latents.clone()
            
            # Clean up
            sampler.pipeline.vae.disable_tiling()
            if sampler_args.cpu_offload:
                # Move VAE components back to CPU after encoding
                sampler.vae.quant_conv.to('cpu')
                sampler.vae.encoder.to('cpu')
        
        print("🎯 Generating video segments...")
        
        # Generate video for each action
        ref_latents = raw_ref_latents
        last_latents = raw_last_latents
        output_videos = []
        
        total_frames = sum(a.frames for a in input_data.actions)
        
        for idx, action in enumerate(input_data.actions):
            is_image = (idx == 0)  # First action starts from image
            
            print(f"🎬 Generating segment {idx+1}/{len(input_data.actions)} with action: {action.id}")
            
            # Generate video segment using the pre-loaded sampler
            outputs = sampler.predict(
                prompt=input_data.prompt,
                action_id=action.id,
                action_speed=action.speed,                    
                is_image=is_image,
                size=input_data.size,
                seed=input_data.seed if input_data.seed is not None else 250160,
                last_latents=last_latents,
                ref_latents=ref_latents,
                video_length=action.frames,
                guidance_scale=input_data.guidance_scale,
                num_images_per_prompt=1,
                negative_prompt=input_data.negative_prompt,
                infer_steps=input_data.infer_steps,
                flow_shift=5.0,
                use_linear_quadratic_schedule=False,
                linear_schedule_end=0.1,
                use_deepcache=True,
                cpu_offload=input_data.cpu_offload,
                ref_images=[raw_ref_image],
                output_dir=str(tmp_path),
                return_latents=True,
                use_sage=False,
            )
            
            # Update latents for next iteration
            ref_latents = outputs["ref_latents"]
            last_latents = outputs["last_latents"]
            output_videos.append(outputs['samples'][0])
        
        # Combine all video segments
        if len(output_videos) == 1:
            final_video = output_videos[0]
        else:
            final_video = torch.cat(output_videos, dim=2)  # Concatenate along time dimension
        
        # Save the final video
        print("💾 Saving final video...")
        video_path = tmp_path / "generated_video.mp4"
        save_videos_grid(final_video, str(video_path), n_rows=1, fps=24)
        
        print(f"✅ Video generation complete! Saved to: {video_path}")
        
        return {
            "video_path": video_path,
            "height": input_data.size[0],
            "width": input_data.size[1], 
            "total_frames": total_frames,
            "log": f"Successfully generated video with {len(input_data.actions)} actions, {total_frames} total frames"
        }