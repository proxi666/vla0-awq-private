import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

# Make a local LeRobot checkout importable for RoboVerse dataset loading.
# There are two layouts we care about:
# 1. Local workspace: <workspace>/vla0/scripts/... and sibling <workspace>/lerobot/src
# 2. Colab clone: /content/vla0/scripts/... and bundled /content/vla0/libs/RoboVerse/libs/lerobot/src
VLA0_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = VLA0_ROOT.parent
LEROBOT_SRC_CANDIDATES = [
    WORKSPACE_ROOT / "lerobot" / "src",
    VLA0_ROOT / "libs" / "RoboVerse" / "libs" / "lerobot" / "src",
]
for lerobot_src in LEROBOT_SRC_CANDIDATES:
    if lerobot_src.exists():
        sys.path.insert(0, str(lerobot_src))
        break

from rv_train.train import (get_cfg, get_dataloader,  # noqa: E402
                            get_pretrained_model)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare VLA-0 action generation across load modes on one dataset sample."
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
        "--split",
        type=str,
        default="train",
        help="Dataset split label to request from get_dataloader. For roboverse this is ignored.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Which sample to compare from the dataset loader.",
    )
    parser.add_argument(
        "--load-modes",
        nargs="+",
        default=["bf16", "int8", "nf4"],
        choices=["bf16", "int8", "nf4", "awq"],
        help="Load modes to compare.",
    )
    parser.add_argument(
        "--generate-temperature",
        type=float,
        default=0.0,
        help="Generation temperature passed into model(..., get_action=True).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save the comparison results as JSON.",
    )
    parser.add_argument(
        "--lerobot-repo-id",
        type=str,
        default="lerobot/libero_10",
        help="LeRobot dataset repo id to use for fetching the comparison sample.",
    )
    return parser.parse_args()


def override_lerobot_settings(
    cfg,
    repo_id: str,
    action_key: str = "action",
    state_key: str = "observation.state",
):
    cfg.defrost()
    base_opts = cfg.DATALOADER.ROBOVERSE.cfg_opts
    overrides = [
        f"LEROBOT.repo_id:{repo_id}",
        f"LEROBOT.action_key:{action_key}",
        f"LEROBOT.state_key:{state_key}",
    ]
    override = ":".join(overrides)
    if base_opts:
        cfg.DATALOADER.ROBOVERSE.cfg_opts = f"{base_opts}:{override}"
    else:
        cfg.DATALOADER.ROBOVERSE.cfg_opts = override
    cfg.freeze()
    return cfg


def get_single_batch(cfg, split: str, sample_index: int, device: str) -> dict[str, Any]:
    dataset = get_dataloader(split=split, cfg=cfg, get_dataset=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )
    if sample_index < 0:
        raise ValueError("sample_index must be >= 0")

    data_batch = None
    for idx, batch in enumerate(loader):
        if idx == sample_index:
            data_batch = batch
            break

    if data_batch is None:
        raise IndexError(f"sample_index {sample_index} is out of range")

    data_batch = loader.dataset.batch_proc(data_batch, device)
    return data_batch


def tensor_to_list(x: torch.Tensor):
    return x.detach().cpu().tolist()


def compare_actions(reference: torch.Tensor, candidate: torch.Tensor):
    diff = candidate - reference
    return {
        "mae": diff.abs().mean().item(),
        "mse": diff.pow(2).mean().item(),
        "max_abs": diff.abs().max().item(),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested, but torch.cuda.is_available() is False"
        )

    model_folder = "/".join(args.checkpoint.split("/")[:-1])
    cfg = get_cfg(f"{model_folder}/config.yaml", "")
    cfg = override_lerobot_settings(cfg, args.lerobot_repo_id)
    batch = get_single_batch(
        cfg=cfg,
        split=args.split,
        sample_index=args.sample_index,
        device=args.device,
    )

    results: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "sample_index": args.sample_index,
        "load_modes": args.load_modes,
        "instruction": batch["instr"][0],
        "modes": {},
    }

    baseline_action = None
    baseline_text = None

    if args.device.startswith("cuda"):
        torch.cuda.set_device(device)

    for load_mode in args.load_modes:
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        model, _ = get_pretrained_model(
            args.checkpoint,
            args.device,
            torch_compile=False,
            load_mode=load_mode,
        )
        model.eval()
        load_seconds = time.perf_counter() - start

        with torch.no_grad():
            output = model(
                **batch,
                get_loss=False,
                get_action=True,
                generate_temperature=args.generate_temperature,
            )

        pred_action = output["out_ori_act"].detach().cpu()
        pred_text = output["pred_action_txt"][0]

        mode_result = {
            "load_seconds": load_seconds,
            "pred_action_txt": pred_text,
            "gt_action_text": output["gt_action_text"][0],
            "first_action_step": tensor_to_list(pred_action[0, 0]),
        }

        if args.device.startswith("cuda"):
            mode_result["cuda_memory_allocated_mb"] = (
                torch.cuda.memory_allocated(device) / 1024**2
            )
            mode_result["cuda_memory_reserved_mb"] = (
                torch.cuda.memory_reserved(device) / 1024**2
            )
            mode_result["cuda_max_memory_allocated_mb"] = (
                torch.cuda.max_memory_allocated(device) / 1024**2
            )

        if baseline_action is None:
            baseline_action = pred_action
            baseline_text = pred_text
            mode_result["vs_bf16"] = None
        else:
            mode_result["vs_bf16"] = compare_actions(baseline_action, pred_action)
            mode_result["exact_text_match_bf16"] = pred_text == baseline_text

        results["modes"][load_mode] = mode_result

        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    output_text = json.dumps(results, indent=2)
    print(output_text)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
