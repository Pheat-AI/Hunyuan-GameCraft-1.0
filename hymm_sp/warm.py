"""
Warm function for pre-loading models into GPU memory.
This gets called during app setup to initialize models once per container.
"""

import torch
from pathlib import Path
import torchvision.transforms as transforms


class ModelWarmer:
    """Handles warming up and managing the pre-loaded models"""
    
    def __init__(self):
        self.hunyuan_video_sampler = None
        self.ref_image_transform = None
        self.device = None
        self.sampler_args = None
        
    def warm_models(self, checkpoint_path: Path, device: torch.device, cpu_offload: bool = False):
        """
        Load and warm up all models into GPU memory.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            device: CUDA device to load models on
            cpu_offload: Whether to enable CPU offloading for memory efficiency
        """
        print("🔥 Warming up models...")
        
        # Import here to avoid circular imports
        from hymm_sp.sample_inference import HunyuanVideoSampler
        from diffusers.hooks import apply_group_offloading
        
        self.device = device
        
        # Create minimal args object for the sampler
        class Args:
            def __init__(self):
                self.cpu_offload = cpu_offload
                self.use_fp8 = False
                self.seed = 250160
                self.rope_theta = 1000000
                self.vae = "hyvae"  # Set default VAE type
                
        args = Args()
        
        # Load the video sampler - this is the expensive operation we want to do once
        print("📥 Loading HunyuanVideoSampler...")
        self.hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
            str(checkpoint_path), 
            args=args, 
            device=torch.device("cpu") if args.cpu_offload else device
        )
        
        # Store the args for later use
        self.sampler_args = self.hunyuan_video_sampler.args
        
        # Enable CPU offloading for memory efficiency if requested
        if args.cpu_offload:
            print("🔄 Setting up CPU offloading...")
            onload_device = torch.device("cuda")
            apply_group_offloading(
                self.hunyuan_video_sampler.pipeline.transformer, 
                onload_device=onload_device, 
                offload_type="block_level", 
                num_blocks_per_group=1
            )
            print("✅ Enabled CPU offloading for transformer blocks")
        
        # Set up image preprocessing transforms (reused for each generation)
        self.ref_image_transform = transforms.Compose([
            self._CropResize((704, 1216)),
            transforms.CenterCrop((704, 1216)),
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] range
        ])
        
        print("✅ Model warming complete! Ready for inference.")
        
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
    
    def get_sampler_args(self):
        """Get the sampler arguments"""
        return self.sampler_args
    
    class _CropResize:
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


# Global instance to be used across the app
model_warmer = ModelWarmer()