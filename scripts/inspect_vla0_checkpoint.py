import argparse
import time

import torch

from rv_train.train import get_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect a local VLA-0 checkpoint load path."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the inference checkpoint folder (model_last) or the old model_last.pth path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device to load on, default cuda:0.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile the loaded model after restore.",
    )
    parser.add_argument(
        "--load-mode",
        type=str,
        default="bf16",
        choices=["bf16", "int8", "nf4", "awq"],
        help="Precision/load mode for the checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested, but torch.cuda.is_available() is False"
        )

    if args.device.startswith("cuda"):
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model, cfg = get_pretrained_model(
        args.checkpoint,
        args.device,
        torch_compile=args.torch_compile,
        load_mode=args.load_mode,
    )
    print(f"load_mode: {args.load_mode}")

    load_seconds = time.perf_counter() - start

    model.eval()

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {args.device}")
    print(f"torch_compile: {args.torch_compile}")
    print(f"exp_model: {cfg.EXP.MODEL}")
    print(f"qwen_model_id: {cfg.MODEL.QWEN.qwen_model_id}")
    print(f"use_lora: {cfg.MODEL.QWEN.use_lora}")
    print(f"use_qlora: {cfg.MODEL.QWEN.use_qlora}")
    print(f"load_seconds: {load_seconds:.3f}")

    if args.device.startswith("cuda"):
        print(
            f"cuda_memory_allocated_mb: {torch.cuda.memory_allocated(device) / 1024**2:.2f}"
        )
        print(
            f"cuda_memory_reserved_mb: {torch.cuda.memory_reserved(device) / 1024**2:.2f}"
        )
        print(
            f"cuda_max_memory_allocated_mb: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f}"
        )


if __name__ == "__main__":
    main()
