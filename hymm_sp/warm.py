"""
Warm function for pre-loading models into GPU memory.
Following the pattern from sample_batch.py
"""

import os
import torch
from pathlib import Path
import torchvision.transforms as transforms
from loguru import logger


class CropResize:
    """Custom transform to resize and crop images to a target size while preserving aspect ratio."""
    def __init__(self, size=(704, 1216)):
        self.target_h, self.target_w = size  

    def __call__(self, img):
        w, h = img.size
        scale = max(self.target_w / w, self.target_h / h)
        new_size = (int(h * scale), int(w * scale))
        resize_transform = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BILINEAR)
        resized_img = resize_transform(img)
        crop_transform = transforms.CenterCrop((self.target_h, self.target_w))
        return crop_transform(resized_img)


class ModelWarmer:
    """Handles warming up and managing the pre-loaded models"""
    
    def __init__(self):
        self.hunyuan_video_sampler = None
        self.ref_image_transform = None
        self.device = None
        self.args = None
        
    def warm_models(self, checkpoint_path: Path, device: torch.device, cpu_offload: bool = False, 
                    use_fp8: bool = False, seed: int = 250160):
        """
        Load and warm up all models into GPU memory.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            device: CUDA device to load models on
            cpu_offload: Whether to enable CPU offloading for memory efficiency
            use_fp8: Whether to use FP8 precision
            seed: Random seed for reproducibility
        """
        logger.info("🔥 Warming up models...")
        
        # Import here to avoid circular imports
        from hymm_sp.sample_inference import HunyuanVideoSampler
        from hymm_sp.config import parse_args
        
        self.device = device
        
        # Use the actual parse_args to get all required attributes
        # Create a minimal command line args list
        cmd_args = [
            "--ckpt", str(checkpoint_path),
            "--seed", str(seed),
            "--video-size", "704", "1216",
            "--infer-steps", "8",  # Use minimal steps for warming
            "--cfg-scale", "1.0",
            "--image-start",
            "--action-list", "w",  # Minimal action
            "--action-speed-list", "0.2",
            "--sample-n-frames", "1",  # Minimal frames
            "--save-path", "/tmp/warm_results",
        ]
        
        if cpu_offload:
            cmd_args.append("--cpu-offload")
        if use_fp8:
            cmd_args.append("--use-fp8")
        
        # Parse args like sample_batch.py does
        import sys
        original_argv = sys.argv
        sys.argv = ["warm.py"] + cmd_args
        self.args = parse_args()
        sys.argv = original_argv  # Restore original argv
        
        # Load the video sampler following sample_batch.py pattern
        logger.info(f"📥 Loading model from checkpoint: {checkpoint_path}")
        self.hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
            self.args.ckpt, 
            args=self.args, 
            device=device if not self.args.cpu_offload else torch.device("cpu")
        )
        
        # Update args with model-specific configurations from the checkpoint
        self.args = self.hunyuan_video_sampler.args
        
        # Enable CPU offloading if specified
        if self.args.cpu_offload:
            logger.info("🔄 Setting up CPU offloading...")
            from diffusers.hooks import apply_group_offloading
            onload_device = torch.device("cuda")
            apply_group_offloading(
                self.hunyuan_video_sampler.pipeline.transformer, 
                onload_device=onload_device, 
                offload_type="block_level", 
                num_blocks_per_group=1
            )
            logger.info("✅ Enabled CPU offloading for transformer blocks")
        
        # Set up image preprocessing transforms (matching sample_batch.py)
        closest_size = (704, 1216)
        self.ref_image_transform = transforms.Compose([
            CropResize(closest_size),
            transforms.CenterCrop(closest_size),
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] range
        ])
        
        logger.info("✅ Model warming complete! Ready for inference.")
        
    def get_sampler(self):
        """Get the pre-loaded video sampler"""
        if self.hunyuan_video_sampler is None:
            raise RuntimeError("Models not warmed up yet. Call warm_models() first.")
        return self.hunyuan_video_sampler
    
    def get_image_transform(self):
        """Get the image preprocessing transform"""
        if self.ref_image_transform is None:
            raise RuntimeError("Models not warmed up yet. Call warm_models() first.")
        return self.ref_image_transform
    
    def get_device(self):
        """Get the device models are loaded on"""
        return self.device
    
    def get_args(self):
        """Get the args object with model configurations"""
        return self.args


# Global instance to be used across the app
model_warmer = ModelWarmer()