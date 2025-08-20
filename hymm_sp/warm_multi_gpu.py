#!/usr/bin/env python
"""
Multi-GPU warming script to be run with torchrun.
This pre-loads models across all GPUs in a distributed manner.
"""

import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path
from loguru import logger

# Add the repo to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hymm_sp.sample_inference import HunyuanVideoSampler
from hymm_sp.config import parse_args
from hymm_sp.modules.parallel_states import initialize_distributed


def warm_multi_gpu():
    """Warm models across multiple GPUs using torchrun"""
    
    # Parse minimal args for warming
    cmd_args = [
        "--ckpt", os.environ.get("CHECKPOINT_PATH", "weights/gamecraft_models/mp_rank_00_model_states_distill.pt"),
        "--seed", "250160",
        "--video-size", "704", "1216",
        "--infer-steps", "8",
        "--cfg-scale", "1.0",
        "--image-start",
        "--action-list", "w",
        "--action-speed-list", "0.2",
        "--sample-n-frames", "1",
        "--save-path", "/tmp/warm_results",
    ]
    
    if os.environ.get("USE_FP8", "false").lower() == "true":
        cmd_args.append("--use-fp8")
    
    # Parse args
    sys.argv = ["warm_multi_gpu.py"] + cmd_args
    args = parse_args()
    
    # Initialize distributed - this expects MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE from torchrun
    initialize_distributed(args.seed)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    logger.info(f"🔥 Warming GPU {rank}/{world_size-1}")
    
    # Set device for this rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Load model on this GPU
    logger.info(f"Loading model on GPU {rank}...")
    sampler = HunyuanVideoSampler.from_pretrained(
        args.ckpt, 
        args=args, 
        device=device
    )
    
    logger.info(f"✅ GPU {rank} warmed successfully!")
    
    # Keep process alive to maintain model in memory
    if os.environ.get("KEEP_ALIVE", "false").lower() == "true":
        logger.info(f"GPU {rank} keeping model warm...")
        while True:
            torch.cuda.synchronize()
            dist.barrier()


if __name__ == "__main__":
    warm_multi_gpu()